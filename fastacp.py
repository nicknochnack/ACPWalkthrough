from typing import List, Dict, Callable, Optional, Union, Any, AsyncGenerator
import importlib.resources
import yaml
import json
import re
# Import SmolAgentsChatMessage and MessageRole from smolagents.models
from smolagents.models import ChatMessage as SmolAgentsChatMessage, MessageRole
from dataclasses import dataclass
from enum import Enum
from colorama import Fore
from acp_sdk.client import Client
from acp_sdk.models import (
    Message,
    MessagePart,
)

# === AgentCollection Implementation ===

class Agent:
    """Representation of an ACP Agent."""
    
    def __init__(self, name: str, description: str, capabilities: List[str]):
        self.name = name
        self.description = description
        self.capabilities = capabilities
    
    def __str__(self):
        return f"Agent(name='{self.name}', description='{self.description}')"


class AgentCollection:
    """
    A collection of agents available on ACP servers.
    Allows users to discover available agents on ACP servers.
    """
    
    def __init__(self):
        self.agents = []
    
    @classmethod
    async def from_acp(cls, *servers) -> 'AgentCollection':
        """
        Creates an AgentCollection by fetching agents from the provided ACP servers.
        
        Args:
            *servers: ACP server client instances to fetch agents from
            
        Returns:
            AgentCollection: Collection containing all discovered agents
        """
        collection = cls()
        
        for server in servers:
            async for agent in server.agents():
                collection.agents.append((server,agent))
        
        return collection
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Find an agent by name in the collection.
        
        Args:
            name: Name of the agent to find
            
        Returns:
            Agent or None: The found agent or None if not found
        """
        # Iterate over the stored (client, agent) tuples
        for client, agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def __iter__(self):
        """Allows iteration over all agents in the collection."""
        return iter(self.agents)


# === ACPCallingAgent Implementation ===

@dataclass
class ToolCall:
    """Represents a tool call with name, arguments, and optional ID."""
    name: str
    arguments: Union[Dict[str, Any], str]
    id: Optional[str] = None


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Logger:
    """Simple logger for agent operations."""
    
    def log(self, content, level=LogLevel.INFO):
        print(f"[{level.value.upper()}] {content}")
    
    def log_markdown(self, content, title=None, level=LogLevel.INFO):
        if title:
            print(f"[{level.value.upper()}] {title}")
        print(f"[{level.value.upper()}] {content}")


class AgentError(Exception):
    """Base class for agent errors."""
    
    def __init__(self, message, logger=None):
        super().__init__(message)
        if logger:
            logger.log(message, level=LogLevel.ERROR)


class AgentParsingError(AgentError):
    """Error when parsing agent output."""
    pass


class AgentToolCallError(AgentError):
    """Error when making a tool call."""
    pass


class AgentToolExecutionError(AgentError):
    """Error when executing a tool."""
    pass


class ActionStep:
    """Represents a step in the agent's reasoning process."""
    
    def __init__(self):
        self.model_input_messages = []
        self.model_output_message = None
        self.model_output = None
        self.tool_calls = []
        self.action_output = None
        self.observations = None


class Tool:
    """Base class for tools that agents can use."""
    
    def __init__(self, name, description, inputs, output_type, client=None):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.output_type = output_type
        self.client = client
    
    async def __call__(self, *args, **kwargs):
        print(Fore.YELLOW + 'Tool being called with args: ' + str(args) + ' and kwargs: ' + str(kwargs) + Fore.RESET)
    
        # Extract the input content from either args or kwargs
        content = ""
        
        # Prioritize 'prompt' or the first input key if defined in tool.inputs
        if self.inputs and "prompt" in self.inputs:
            content = kwargs.get("prompt", "")
        elif self.inputs and list(self.inputs.keys()):
            first_input_key = list(self.inputs.keys())[0]
            content = kwargs.get(first_input_key, "")
        elif args and isinstance(args[0], str): # Fallback for positional string argument
            content = args[0]
        elif kwargs: # Generic fallback, but less robust
            content = next(iter(kwargs.values()))

        if not content and self.inputs: # If content is still empty, check if any input is required
            required_inputs = [k for k, v in self.inputs.items() if v.get('required', False)]
            if required_inputs:
                self.logger.log(f"Warning: Tool '{self.name}' called without required input: {required_inputs}", level=LogLevel.WARNING)
            
        # Now use the extracted content in your message
        print(Fore.MAGENTA + content + Fore.RESET) 

        # ACP client expects 'input' to be a list of Message objects
        response = await self.client.run_sync(
            agent=self.name, 
            input=[Message(parts=[MessagePart(content=content, content_type="text/plain")])]
        )

        print(Fore.RED + str(response) + Fore.RESET) 

        # ACP client returns outputs, typically as a list of Message objects
        return response.output[0].parts[0].content


class MultiStepAgent:
    """Base class for agents that operate in multiple steps."""
    
    def __init__(
        self,
        tools: Dict[str, Tool],
        model: Callable,
        prompt_templates: Dict[str, str],
        planning_interval: Optional[int] = None,
        **kwargs
    ):
        self.tools = tools
        self.model = model
        self.prompt_templates = prompt_templates
        self.planning_interval = planning_interval
        self.managed_agents = kwargs.get("managed_agents", {})
        self.logger = Logger()
        self.state = {}
        # input_messages now stores SmolAgentsChatMessage objects
        self.input_messages: List[SmolAgentsChatMessage] = []
    
    def initialize_system_prompt(self) -> str:
        """Generate the system prompt for the agent."""
        raise NotImplementedError
    
    def write_memory_to_messages(self) -> List[SmolAgentsChatMessage]:
        """
        Converts the internal message representation to a list of SmolAgentsChatMessage objects,
        ensuring content is formatted as list[dict[str, Any]] for text.
        This acts as a safety net before passing messages to the model.
        """
        cleaned_messages = []
        for msg in self.input_messages:
            if isinstance(msg, SmolAgentsChatMessage):
                # Ensure content is in the expected list format for SmolAgentsChatMessage
                if isinstance(msg.content, str):
                    msg.content = [{"type": "text", "text": msg.content}]
                elif msg.content is None:
                    msg.content = []
                elif isinstance(msg.content, list):
                    for i, item in enumerate(msg.content):
                        if isinstance(item, str):
                            msg.content[i] = {"type": "text", "text": item}
                        elif not isinstance(item, dict) or "type" not in item or "text" not in item:
                            self.logger.log(f"Warning: Malformed content part in ChatMessage: {item}", level=LogLevel.WARNING)
                cleaned_messages.append(msg)
            elif isinstance(msg, dict):
                # Attempt to convert dict to SmolAgentsChatMessage (fallback)
                try:
                    role_enum = MessageRole(msg.get("role"))
                    content_val = msg.get("content")
                    if isinstance(content_val, str):
                        content_val = [{"type": "text", "text": content_val}]
                    elif content_val is None:
                        content_val = []

                    cleaned_messages.append(
                        SmolAgentsChatMessage(
                            role=role_enum,
                            content=content_val,
                            tool_calls=msg.get("tool_calls")
                        )
                    )
                except ValueError as e:
                    self.logger.log(f"Error converting message dict to SmolAgentsChatMessage: {msg} - {e}", level=LogLevel.ERROR)
                    cleaned_messages.append(msg)
            else:
                self.logger.log(f"Unexpected message type in input_messages: {type(msg)} - {msg}", level=LogLevel.ERROR)
                cleaned_messages.append(msg)

        self.input_messages = cleaned_messages # Update the list with cleaned messages
        return self.input_messages
    
    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """Perform one step in the agent's reasoning process."""
        raise NotImplementedError


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    """Helper function to populate a template with variables."""
    result = template
    for key, value in variables.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    return result


class ACPCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like ACP agent calls, similarly to how ToolCallingAgent uses tool calls,
    but directed at remote ACP agents instead of local tools.
    
    Args:
        acp_agents (`dict[str, Agent]`): ACP agents that this agent can call.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`Dict[str, str]`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        # Corrected type hint for acp_agents to reflect {'agent':Agent, 'client':Client} structure
        acp_agents: Dict[str, Dict[str, Any]],
        # Corrected type hint for model to expect SmolAgentsChatMessage
        model: Callable[[List[SmolAgentsChatMessage]], SmolAgentsChatMessage],
        prompt_templates: Optional[Dict[str, str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        # Default prompt templates if none provided
        if prompt_templates is None:
            prompt_templates = {
                "system_prompt": """You are a supervisory agent that can delegate tasks to specialized ACP agents.
                Available agents:
                {agents}

                Your task is to:
                1. Analyze the user's request
                2. Call the appropriate agent(s) to gather information
                3. When you have a complete answer, ALWAYS call the final_answer tool with your response
                4. Do not provide answers directly in your messages - always use the final_answer tool
                5. If you have sufficient information to complete a task do not call out to another agent unless required

                Remember:
                - Always use the final_answer tool when you have a complete answer
                - Do not provide answers in your regular messages
                - Chain multiple agent calls if needed to gather all required information
                - The final_answer tool is the only way to return results to the user
                """
            }
        
        # Convert ACP agents to a format similar to tools
        acp_tools = {}

        # Iterate over agent_info dict, which contains 'agent' and 'client'
        for name, agent_info in acp_agents.items():
            agent_obj = agent_info['agent'] # Access the Agent object
            client_obj = agent_info['client'] # Access the Client object
            
            acp_tools[name] = Tool(
                name=name,
                description=agent_obj.description, # Access description from Agent object
                inputs={"prompt": {"type":"string","description":f"The prompt to pass to the {name} agent"}},
                output_type="str",
                client=client_obj # Pass the correct client object
            )
        
        # Add final_answer tool
        acp_tools["final_answer"] = Tool(
            name="final_answer",
            description="Provide the final answer to the user's request",
            # Corrected inputs format for final_answer tool to match JSON schema expectation
            inputs={"answer": {"type":"string", "description": "The final answer to provide to the user"}},
            output_type="str"
        )
        
        async def final_answer_func(answer: str, **kwargs): # Added type hint for clarity
            return answer
        
        acp_tools["final_answer"].__call__ = final_answer_func
        
        super().__init__(
            tools=acp_tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        
        self.acp_agents = acp_agents
    
    def initialize_system_prompt(self) -> str:
        """Generate the system prompt for the agent with ACP agent information."""
        # Access agent description correctly from agent_info dict
        agent_descriptions = "\n".join(
            [f"- {name}: {agent_info['agent'].description}" for name, agent_info in self.acp_agents.items()]
        )
        # Add final_answer tool description to prompt
        final_answer_tool_desc = self.tools['final_answer'].description
        agent_descriptions += f"\n- final_answer: {final_answer_tool_desc}"
        
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"agents": agent_descriptions},
        )
        return system_prompt

    def save_to_memory(self, key: str, value: Any) -> None:
        """Save a value to the agent's persistent memory."""
        self.state[key] = value
        self.logger.log(f"Saved to memory: {key}={value}", level=LogLevel.DEBUG)



    async def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the reasoning process: the agent thinks, calls ACP agents, and observes results.
        Returns None if the step is not final.
        """
        # Use write_memory_to_messages to get cleaned messages, which ensures SmolAgentsChatMessage objects
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages.copy()
        
        try:
            import logging
            logging.warning(f"Tools being passed to model: {[tool.name for tool in list(self.tools.values())[:-1]]}")
            # Ensure model receives SmolAgentsChatMessage objects, which write_memory_to_messages now guarantees
            model_message: SmolAgentsChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values())[:-1],
                stop_sequences=["Observation:", "Calling agents:"],
            )
            memory_step.model_output_message = model_message
        except Exception as e:
            # Improved error logging to show message format accurately for SmolAgentsChatMessage
            error_details_message_format = [msg.dict() if hasattr(msg, 'dict') else { 'role': str(msg.role), 'content': msg.content, 'tool_calls': msg.tool_calls} for msg in memory_messages]
            error_details_tools = [tool.name for tool in list(self.tools.values())[:-1]]
            self.logger.log(f"Error details - message format: {error_details_message_format}", level=LogLevel.ERROR)
            self.logger.log(f"Error details - tools: {error_details_tools}", level=LogLevel.ERROR)
            raise AgentParsingError(f"Error while generating or parsing output:\n{e}", self.logger) from e
        
        self.logger.log_markdown(
            content=model_message.content if model_message.content else str(model_message.raw),
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )
        
        # Unified tool call parsing logic
        if not (hasattr(model_message, 'tool_calls') and model_message.tool_calls):
            # If no tool calls, treat content as final answer or try to parse implicit tool call
            if model_message.content:
                # Content from model could be a list of dicts. Extract text.
                extracted_content_text = ""
                if isinstance(model_message.content, list) and model_message.content and "text" in model_message.content[0]:
                    extracted_content_text = model_message.content[0]["text"]
                elif isinstance(model_message.content, str): # Fallback if model gives raw string unexpectedly
                    extracted_content_text = model_message.content
                else:
                    extracted_content_text = str(model_message.content) # Convert to string for general handling

                if "final_answer" in extracted_content_text.lower():
                    self.logger.log(
                        f"Final answer detected in content: {extracted_content_text}",
                        level=LogLevel.INFO,
                    )
                    memory_step.action_output = extracted_content_text
                    return extracted_content_text
                else:
                    # Attempt to parse tool call from raw content using regex
                    # Ensure 're' is imported at the top of the file
                    tool_match = re.search(r'(?:tool|agent):\s*(\w+)\s*(?:arguments:\s*(.*))?', extracted_content_text, re.IGNORECASE | re.DOTALL)
                    if tool_match:
                        agent_name = tool_match.group(1)
                        raw_args = tool_match.group(2)
                        agent_arguments = {}
                        if raw_args:
                            try:
                                agent_arguments = json.loads(raw_args.strip())
                            except json.JSONDecodeError:
                                agent_arguments = {"prompt": raw_args.strip()}
                        elif agent_name in self.tools and "prompt" in self.tools[agent_name].inputs:
                            # If no explicit arguments provided, use remaining content as prompt.
                            # This needs to be robust for various model output styles.
                            remaining_content_start = extracted_content_text.find(tool_match.group(0)) + len(tool_match.group(0))
                            remaining_content = extracted_content_text[remaining_content_start:].strip()
                            if remaining_content:
                                agent_arguments = {"prompt": remaining_content}

                        memory_step.model_output = str(f"Called Agent: '{agent_name}' with arguments: {agent_arguments}")
                        memory_step.tool_calls = [ToolCall(name=agent_name, arguments=agent_arguments, id="synthetic_id")]
                        return await self._process_tool_call(memory_step, agent_name, agent_arguments)
            
            raise AgentParsingError(
                "Model did not call any agents and no final answer detected. Content: " + (str(model_message.content) or "None"), 
                self.logger
            )
        
        # Standardized access for SmolAgentsChatMessageToolCall
        tool_call = model_message.tool_calls[0] # This is a SmolAgentsChatMessageToolCall object
        
        agent_name = tool_call.function.name
        agent_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id # Directly access .id

        memory_step.model_output = str(f"Called Agent: '{agent_name}' with arguments: {agent_arguments}")
        memory_step.tool_calls = [ToolCall(name=agent_name, arguments=agent_arguments, id=tool_call_id)]
        
        return await self._process_tool_call(memory_step, agent_name, agent_arguments)
        
    async def _process_tool_call(self, memory_step: ActionStep, agent_name: str, agent_arguments: Any) -> Union[None, Any]:
        """
        Process a tool call with the given name and arguments.
        """
        
        # Execute the tool call
        self.logger.log(
            f"Calling agent: '{agent_name}' with arguments: {agent_arguments}",
            level=LogLevel.INFO,
        )
        
        if agent_name == "final_answer":
            # Handle the final answer
            if isinstance(agent_arguments, dict):
                answer = agent_arguments.get("answer", agent_arguments) # Use .get for safety
            else:
                answer = agent_arguments
            
            # If answer refers to a state variable, use that
            if isinstance(answer, str) and answer in self.state:
                final_answer = self.state[answer]
                self.logger.log(
                    f" Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    f"Final result: {final_answer}", # Changed log message for clarity
                    level=LogLevel.INFO,
                )
            
            memory_step.action_output = final_answer
            return final_answer
        else:
            # Execute the ACP agent call
            if agent_arguments is None:
                agent_arguments = {}
            
            # Removed sanitize_inputs_outputs as it's not a universal kwarg
            observation = await self.execute_tool_call(agent_name, agent_arguments)
            updated_information = str(observation).strip()

            self.save_to_memory(f"{agent_name}_response", updated_information)
            
            self.logger.log(
                f"Observations: {updated_information}",
                level=LogLevel.INFO,
            )
            
            memory_step.observations = updated_information
            return None
    
    def _substitute_state_variables(self, arguments: Union[Dict[str, str], str]) -> Union[Dict[str, Any], str]:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        # Added handling for string arguments directly mapping to state
        if isinstance(arguments, str):
            return self.state.get(arguments, arguments)
        return arguments
    
    async def execute_tool_call(self, agent_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute an ACP agent call with the provided arguments.
        
        Args:
            agent_name (`str`): Name of the ACP agent to call.
            arguments (dict[str, str] | str): Arguments passed to the agent call.
        """
        # Check if the agent exists
        available_tools = {**self.tools}
        if agent_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown agent {agent_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )
        
        # Get the tool and substitute state variables in arguments
        tool = available_tools[agent_name]
        arguments = self._substitute_state_variables(arguments)
        
        try:
            # Call agent with appropriate arguments
            if isinstance(arguments, dict):
                # Removed sanitize_inputs_outputs here
                return await tool(**arguments)
            elif isinstance(arguments, str):
                # Check for 'prompt' input for string arguments
                if tool.inputs and "prompt" in tool.inputs:
                    return await tool(prompt=arguments)
                else:
                    return await tool(arguments)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")
                
        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            error_msg = (
                f"Invalid call to agent '{agent_name}' with arguments {json.dumps(arguments)}: {e}\n"
                "You should call this agent with correct input arguments.\n"
                f"Expected inputs: {json.dumps(tool.inputs)}\n"
                f"Returns output type: {tool.output_type}\n"
                f"Agent description: '{description}'"
            )
            raise AgentToolCallError(error_msg, self.logger) from e
            
        except Exception as e:
            # Handle execution errors
            error_msg = (
                f"Error executing agent '{agent_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                "Please try again or use another agent"
            )
            raise AgentToolExecutionError(error_msg, self.logger) from e

    async def run(self, query: str, max_steps: int = 10) -> str:
        """
        Run the agent to completion with a user query.
        
        Args:
            query (str): The user's query or request
            max_steps (int): Maximum number of steps before giving up, default 10
            
        Returns:
            str: Final answer from the agent
        """
        # Initialize memory with the user query in the correct format for LiteLLM
        # Initialize messages as SmolAgentsChatMessage objects with list content
        user_message = SmolAgentsChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": query}])
        system_prompt_text = self.initialize_system_prompt()
        system_message = SmolAgentsChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": system_prompt_text}])
        self.input_messages = [system_message, user_message]
        
        # Run steps until we get a final answer or hit max steps
        result = None
        for step_num in range(max_steps):
            self.logger.log(f"Step {step_num + 1}/{max_steps}", level=LogLevel.INFO)

            # Add memory context to the messages if we have any state
            if self.state and step_num > 0:
                memory_context_text = "Current memory state:\n" # Renamed to avoid confusion with the dict
                for key, value in self.state.items():
                    memory_context_text += f"- {key}: {value}\n"
                
                # Append new messages as SmolAgentsChatMessage objects with list content
                self.input_messages.append(SmolAgentsChatMessage(
                    role=MessageRole.SYSTEM,
                    content=[{"type": "text", "text": memory_context_text}]
                ))
            
            # Create a new action step and execute it
            memory_step = ActionStep()
            
            try:
                result = await self.step(memory_step)
                
                # If we got a final result, return it
                if result is not None:
                    return result
                
                # Otherwise, add observation to input messages for next step
                if hasattr(memory_step, 'observations') and memory_step.observations:
                    # Add assistant message if it exists
                    if hasattr(memory_step, 'model_output_message') and memory_step.model_output_message:
                        content_from_model = memory_step.model_output_message.content
                        # Ensure content_from_model is formatted as list[dict] if it's text
                        if isinstance(content_from_model, str):
                            content_from_model = [{"type": "text", "text": content_from_model}]
                        elif content_from_model is None:
                            content_from_model = [] # Default to empty list for None content

                        if content_from_model:
                            self.input_messages.append(SmolAgentsChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=content_from_model
                            ))
                    
                    # Add observation as user message
                    # Ensure observation message content is list[dict]
                    self.input_messages.append(SmolAgentsChatMessage(
                        role=MessageRole.USER, 
                        content=[{"type": "text", "text": f"Observation: {memory_step.observations}"}]
                    ))
            except Exception as e:
                self.logger.log(f"Error in step {step_num + 1}: {str(e)}", level=LogLevel.ERROR)
                # Add error message to conversation
                # Ensure error message content is list[dict]
                self.input_messages.append(SmolAgentsChatMessage(
                    role=MessageRole.SYSTEM, # Changed to system role for error messages
                    content=[{"type": "text", "text": f"Error occurred: {str(e)}. Please try a different approach or provide a final answer."}]
                ))
        
        # If we hit max steps without a final answer
        return "I wasn't able to complete this task within the maximum number of steps."