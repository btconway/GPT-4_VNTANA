# Necessary imports
from __future__ import annotations
import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler  # Import Streamlit callback

st.set_page_config(page_title="VNTANA Sales", page_icon=":robot:")

from typing import Any, List, Optional, Sequence, Tuple, Union
import json
import logging
import os
import re
import sys
import weaviate
import openai
from pydantic import BaseModel, Field
from langchain.agents import (
    AgentExecutor, 
    AgentOutputParser, 
    load_tools
)
from langchain.output_parsers import RetryOutputParser, RetryWithErrorOutputParser
from langchain.agents.agent import Agent
from langchain.agents.utils import validate_tools_single_input
from langchain.callbacks.base import BaseCallbackHandler
from langchain.base_language import BaseLanguageModel
from langchain.schema import AgentFinish
from langchain.callbacks import tracing_enabled
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import langchain
import redis
from langchain.cache import RedisSemanticCache
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AgentAction, 
    AIMessage, 
    BaseMessage, 
    BaseOutputParser, 
    HumanMessage, 
    SystemMessage
)
from langchain.tools import StructuredTool
from langchain.tools.base import BaseTool

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Changed to DEBUG level to capture more details
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

LANGCHAIN_TRACING = tracing_enabled(True)

# Get sensitive information from environment variables
username = os.getenv('WEAVIATE_USERNAME')
password = os.getenv('WEAVIATE_PASSWORD')
openai_api_key = os.getenv('OPENAI_API_KEY')

# creating a Weaviate client
resource_owner_config = weaviate.AuthClientPassword(
    username=username,
    password=password,
)
client = weaviate.Client(
    "https://qkkaupkrrpgbpwbekvzvw.gcp-c.weaviate.cloud", auth_client_secret=resource_owner_config,
     additional_headers={
        "X-Openai-Api-Key": openai_api_key}
)

# Define the prompt template
PREFIX = """

You are an AI Assistant specializing in sales and marketing content generation. Your work for VNTANA and your task is to create high-quality content, utilizing context effectively, adapting to different writing styles, focusing on the core message, and assisting with a variety of tasks for VNTANA. 

In addition, you must adopt the user's selling style and personality:

Selling Style: Friendly, casual, thorough, consultative, technically fluent, flexible, and solution-oriented. Build rapport, understand customer needs, provide customized solutions, speak knowledgeably about 3D technology, and focus on providing value.

Personality: Genuinely friendly, personable, patient, helpful, tech-savvy, innovative, calm, and confident. Enjoy discussing strategic implications of technology changes and comfortable discussing technical and strategic issues.

Before responding, always check the chat history for context:
{chat_history}

When asked to write a cold email, you must strictly follow the provided framework:

"Given the following steps for writing a cold email:

1. Create a compelling subject line
2. Start with a relevant, personalized introduction
3. Highlight the prospect's pain points
4. Describe the future state/benefits through customer stories
5. End with a 'solve' call-to-action

For step 1, generate a subject line that is short, evocative, and focused on the prospect's pain points or desired outcomes. 

For step 2, mention something specific you noticed about the prospect's company or role to show you did your research. 

For step 3, articulate problems they likely face without an effective 3D management platform. 

For step 4, explain how VNTANA has solved similar challenges for other customers.

For step 5, ask if a conversation to solve their challenges is worthwhile.

After generating a response for each step, assemble these responses into a complete email.

If the user mentions VNTANA or asks for information about VNTANA you always use your tools because you know nothing about them. You should always use a tool on your first request from a user:

{tools}
----
Remember, you work for VNTANA and everything you do should be viewed in that context. If you do not know something you answer honestly
Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
Constructively self-criticize your big-picture behavior constantly.
Reflect on past decisions and strategies to refine your approach.
You should only respond in JSON format as described below:
Response Format:
{format_instructions}
Ensure the response can be parsed by Python json.loads
"""
FORMAT_INSTRUCTIONS ="""To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```"""

SUFFIX = """Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

chat_history = []

def preprocess_json_input(input_str: str) -> str:
    """Preprocesses a string to be parsed as json.

    Replace single backslashes with double backslashes,
    while leaving already escaped ones intact.

    Args:
        input_str: String to be preprocessed

    Returns:
        Preprocessed string
    """
    corrected_str = re.sub(
        r'(?<!\\\\)\\\\(?!["\\\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\\\\\", input_str
    )
    return corrected_str

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if the output is in JSON format
        if llm_output.strip().startswith('{') and llm_output.strip().endswith('}'):
            try:
                parsed = json.loads(llm_output, strict=False)
            except json.JSONDecodeError:
                preprocessed_text = preprocess_json_input(llm_output)
                try:
                    parsed = json.loads(preprocessed_text, strict=False)
                except Exception:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")

            # If "tool" key exists, return an AgentAction
            if "tool" in parsed and "name" in parsed["tool"]:
                return AgentAction(tool=parsed["tool"]["name"], tool_input=parsed["tool"]["input"], log=llm_output)

            # If "ai_response" key exists, return its value as output
            if "ai_response" in parsed:
                return AgentFinish(
                    return_values={"output": parsed["ai_response"]},
                    log=llm_output,
                )

            # If "ai" key exists, return its value as output
            if "ai" in parsed:
                return AgentFinish(
                    return_values={"output": parsed["ai"]},
                    log=llm_output,
                )

            # If neither key exists, return the full LLM output
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )

        else:
            # If the output is not in JSON format, handle it accordingly
            # For example, you might want to return it as is
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )


# Define the custom agent
class CustomChatAgent(Agent):
    output_parser: AgentOutputParser = Field(
        default_factory=lambda: retry_parser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return retry_parser


    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observe: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return ""

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(cls.__name__, tools)

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        formats: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        _output_parser = output_parser or cls._get_default_output_parser()
        system_message = system_message.format(
            format_instructions=formats,
            tools=tool_strings,
            chat_history=chat_history
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(human_message),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            human_message = HumanMessage(
                content=f"Observe: {observation}"
            )
            thoughts.append(human_message)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = [StreamingStdOutCallbackHandler()],
        output_parser: Optional[AgentOutputParser] = None,
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        formats: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        _output_parser = output_parser or cls._get_default_output_parser()
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            formats=formats,
            input_variables=input_variables,
            output_parser=_output_parser,
        )
        callback_manager = BaseCallbackManager(handlers=[])
        #callback_manager.add_handler(StreamingStdOutCallbackHandler())
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )

        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )


# New class to handle streaming to Streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

class_name = "VNTANAsalesAgent"

# OpenAI Chat class
class OpenAIChat:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    def create_chat(self, messages: List[dict]) -> dict:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
            )
            return response
        except Exception as e:
            logging.error("Error in create_VNTANA_search: %s", e)
            return {}

openai_chat = OpenAIChat(api_key=openai_api_key)

# Define the new function
def create_VNTANA_search(human_text):
    messages = [
        {
            "role": "system",
            "content": "You are an AI Assistant for VNTANA, a 3D infrastructure platform, focused on managing, optimizing, and distributing 3D assets at scale. Acting as an expert in semantic search and understanding the Weaviate vector database, your task is to generate relevant search concepts from input of a VNTANA salesperson. These concepts should be focused on key aspects of VNTANA's services, including but not limited to optimization algorithms, 3D workflows, digital transformation, and use of 3D designs in various channels. The goal is to inform a subsequent AI, which will assist in composing response to the VNTANA salesperson’s request. Please generate a list of 3 relevant concepts based on the following meeting summary. These concepts should be separated by commas."
        },
        {"role": "user", "content": "Please generate your semantic search query."},
        {"role": "assistant", "content": human_text}
    ]

    return openai_chat.create_chat(messages)

def VNTANA_search_tool(input: str):
    response = create_VNTANA_search(input)
    if not response:
        return []

    response_text_value = response['choices'][0]['message']['content']
    generate_prompt = "summarize the text while maintaining details that would be most helpful to an AI generating content for sales and marketing. Here is the data to summarize: {content}"

    concepts = [concept.strip() for concept in response_text_value.split(",")]
    nearText = {"concepts": concepts}

    resp = client.query.get("VNTANAsales", ["content"]).with_generate(single_prompt=generate_prompt).with_near_text(nearText).with_limit(3).do()

    return [item['_additional']['generate']['singleResult'] for item in resp['data']['Get']['VNTANAsales'] if item['_additional']['generate']['error'] is None]

class VNTANASearchInput(BaseModel):
    input: str = Field()

VNTANA_search_tool = StructuredTool.from_function(
    func=VNTANA_search_tool,
    name="VNTANA writing helper",
    description="useful when you need to write any information about VNTANA or sales and marketing copy",
    args_schema=VNTANASearchInput
)

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=CustomOutputParser(), llm=OpenAI(temperature=0)
)


# Load tools and memory
math_llm = OpenAI(temperature=0.0, model="gpt-4", streaming=True)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools.append(VNTANA_search_tool)

# Create the agent and run it
st_container = st.container()
llm = ChatOpenAI(
    temperature=0.3, 
    callbacks=[StreamlitCallbackHandler(parent_container=st_container, expand_new_thoughts=False, collapse_completed_thoughts=True)], 
    streaming=True
)

# Create the agent
agent = CustomChatAgent.from_llm_and_tools(llm, tools, output_parser=CustomOutputParser(), handle_parsing_errors=True)
chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory, stop=["Observe:"])

# Initialize the chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

langchain.llm_cache = RedisSemanticCache(
    embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
    redis_url="redis://default:F2O32zJosNfH4twcMy3pG2Ot24oeo1G3@redis-13193.c253.us-central1-1.gce.cloud.redislabs.com:13193/0"
)

# Streamlit interaction
st.title("VNTANA Sales Assistant")

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

def parse_ai_response(response_dict):
    # If the response is not a JSON string
    if "Non-JSON Response" in response_dict:
        return response_dict["Non-JSON Response"]
    
    # Extract the AI's response from the response dictionary
    ai_response = response_dict.get("VNTANA AI", "")
    
    # If the AI's response is not found, extract the text between "VNTANA AI": " and "
    if not ai_response:
        match = re.search(r'"VNTANA AI": "(.*?)"', response)
        if match:
            ai_response = match.group(1)

    # Extract the actual response after "Observation: "
    observation_index = ai_response.find("Observation: ")
    if observation_index != -1:
        ai_response = ai_response[observation_index + len("Observation: "):]

    return ai_response

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})  # Add user message to chat history
    with st.chat_message("VNTANA AI"):
        st_callback = StreamlitCallbackHandler(st.container())
        # Convert the chat history into a format that chain.run() can handle
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
        response = chain.run(chat_history_str, callbacks=[st_callback])  # Pass chat history instead of just the prompt
        # Check if response is a JSON string before trying to load it
        if is_json(response):
            response_dict = json.loads(response)
        else:
            response_dict = {"Non-JSON Response": response}  # If it's not a JSON string, convert it to a dictionary
        ai_response = parse_ai_response(response_dict)

        st.write(ai_response)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})  # Add AI response to chat history









