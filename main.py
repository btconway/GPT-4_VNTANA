# Necessary imports
from __future__ import annotations
import streamlit as st
from langchain.callbacks.streamlit import StreamlitCallbackHandler  # Import Streamlit callback

st.set_page_config(page_title="VNTANA Sales", page_icon="Profile_Avatar.jpg")
st.sidebar.image("Profile_Avatar.jpg")
st.info("`I am an AI that can help you generate sales and marketing content. For example, I can write email sequences, generate marketing copy, and more."
    "To get the best results, include VNTANA in your request and I can access VNTANA's vector database. You can also say somethign like 'use your tools'`")
st.cache_resource.clear()


from typing import Any, List, Optional, Sequence, Tuple, Union, Type
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
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
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
#from langchain.cache import RedisSemanticCache
from langchain.llms import OpenAI
from langchain.memory import ConversationTokenBufferMemory
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
from langchain.tools.base import BaseTool

logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Changed to DEBUG level to capture more details
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

LANGCHAIN_TRACING = tracing_enabled(True)


# Get sensitive information from environment variables
username = os.getenv('WEAVIATE_USERNAME')
password = os.getenv('WEAVIATE_PASSWORD')
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')


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

You are an AI Assistant specializing in sales and marketing content generation. Your work for VNTANA and your task is to create high-quality content, utilizing context effectively, focusing on the core message as well customer pains, and assisting with a variety of tasks for VNTANA. VNTANA is a 3D infrastructure platform that enables brands to easily manage, optimize, and distribute 3D assets at scale, offering automated 3D optimization tools that reduce file sizes up to 99% while maintaining high visual fidelity for deployment across web, mobile, social media, AR, VR, and metaverse. Trusted by leading brands, VNTANA streamlines 3D workflows to accelerate digital transformation initiatives from design to commerce.

You always adopt "the challenger method" of selling as our product is new and customers may not understand e extent to which VNTANA's product could benefit them. Keep this top of mind as you write copy. Here is the challenger style of selling:
"- Highlight the problem of inefficient 3D asset management. Explain challenges brands face trying to prepare design files for use across web, mobile, AR, VR, metaverse with siloed solutions. Emphasize pain points like manual processing, quality issues, delayed time to market. 

- Show how VNTANA is the solution to these problems. Explain the platform's benefits like automated 3D optimization to reduce file sizes up to 99% without quality loss, ability to instantly convert design files into usable formats, headless API integration to connect with existing infrastructure. Give examples of specific features like bulk upload, configurable pipelines, plugins.

- Customize messaging for the prospect's needs. Ask questions to understand their current workflows, bottlenecks, and goals. Tailor content to address their specific use cases and objectives. Reference client case studies in their industry when possible.

- Take control of the narrative. Educate prospects on importance of 3D to stay competitive. Assert VNTANA's unique expertise in spatial computing, computer vision and 3D infrastructure. Highlight patents, leadership team's experience. 

- Convey urgency and value. Explain why upgrading 3D infrastructure now is crucial to accelerating digital transformation. Quantify VNTANA's impact - faster time to market, increased sales, lower costs and carbon footprint. Push prospects to action.

- Maintain consultative tone throughout. Avoid overt selling. Pose thoughtful questions, listen carefully, and offer personalized recommendations. Keep prospect's best interest top of mind."

Adopt this personality:

Personality: Genuinely friendly but not salesy, direct, personable, informal, uses casual but professional language, very succint,patient, helpful, tech-savvy, innovative, calm, and confident.Comfortable discussing technical and strategic issues.

Before responding, always check the chat history for context:
{chat_history}

You will sometimes receive information on how to write cold emails or other sales techniques. It's important that you follow that information but always keep any emails your write to no more than 500 characters.

If the user mentions VNTANA, asks for information about VNTANA, or the task appears to be sales and marketing related and may benefit from some additional resources you always use your tools because you know nothing about VNTANA. You should always use a tool on your first request from a user:

{tools}
----
Remember, you work for VNTANA and everything you do should be viewed in that context. If you do not know something you answer honestly. NEVER make up any client names. Keep any email you are asked to write under 500 words.
Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
Constructively self-criticize your big-picture behavior constantly.
Reflect on past decisions and strategies to refine your approach.
When you decide to use a tool, pass the entire user input to the tool as it has its own intelligence and more context is helpful to the tool.
You should only respond in the format as described below:

Response Format:
{format_instructions}
"""
FORMAT_INSTRUCTIONS ="""To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

Whenever you use a tool, you must wait until your receive the results of the tool before responding to the Human.When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
"AI:" [your response here]
```"""

SUFFIX = """Begin!
REMEMBER, you must ALWAYS follow your FORMAT INSTRUCTIONS when responding to the Human.
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
        logging.info("Starting parsing of LLM output")

        # Check if the output contains the prefix "AI:"
        if "AI:" in llm_output:
            logging.info("Detected prefix 'AI:' in LLM output")
            return AgentFinish(
                return_values={"output": llm_output.split("AI:")[-1].strip()},
                log=llm_output,
            )
        logging.info(AgentFinish)

        # If the prefix is not found, use a regular expression to extract the action and action input
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output)
        if match:
            action = match.group(1)
            action_input = match.group(2)
            logging.info(f"Match found. Action: {action.strip()}, Action Input: {action_input.strip(' ')}")
            return AgentAction(action.strip(), action_input.strip(' '), llm_output)
        logging.info(AgentAction)
        logging.info("No prefix 'AI:' or match found. Returning full LLM output.")
        # If neither condition is met, return the full LLM output
        return AgentFinish(
            return_values={"output": llm_output},
            log=llm_output,
        )


retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=CustomOutputParser(), llm=OpenAI(temperature=0)
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

class_name = "VNTANAsalesAgent2"

class VNTANAsalesQuerySchema(BaseModel):
    query: str = Field(description="should be a search query")

class VNTANAsalesQueryTool(BaseTool):
    name = "VNTANA Search Tool"
    description = "useful whenever writing copy for sales and marketing or looking for information about VNTANA"
    args_schema: Type[VNTANAsalesQuerySchema] = VNTANAsalesQuerySchema

    def truncate_response(self, response: str, max_length: int = 2500) -> str:
        """Truncate the response if it exceeds the max_length."""
        if len(response) > max_length:
            return response[:max_length]
        return response

    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        results = []  # Initialize an empty list to store the results
        try:
            weaviate_query = query_weaviate(query)
            logging.info(weaviate_query)
            if weaviate_query is not None:
                concept = weaviate_query  # Split the query into individual concepts
                nearText = {"concepts": [concept]}  # Search for each concept individually
                resp = client.query.get(class_name, ["content"]).with_near_text(nearText).with_limit(4).do()
                resp = self.truncate_response(resp)  # Truncate the response if it exceeds 3000 characters
                results.append(resp)
                logging.info(resp)  # Changed from print to logging.info
        except Exception as e:
            logging.error(f"Error occurred while querying: {e}")
            raise e
        return {"results": results}  # Return the results as a dictionary

    def _arun(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        pass  # Dummy implementation

def query_weaviate(input):
    try:
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
                {"role": "system", "content": """You are an AI Assistant for VNTANA, a 3D infrastructure platform, focused on managing, optimizing, and distributing 3D assets at scale. Acting as an expert in semantic search and understanding the Weaviate vector database, your task is to generate relevant search concepts from input of a VNTANA salesperson. These concepts should be focused on key aspects of VNTANA's services, including but not limited to optimization algorithms, 3D workflows, digital transformation, and use of 3D designs in various channels. The goal is to inform a subsequent AI, which will assist in composing response to the VNTANA salesperson’s request. Please generate a list of up to 4 relevant concepts that will be helpful to search in order to answer the user's query. If the user requests a specific type of content, then one of your concepts should be that type of content. For example, if the user said, 'Write me a cold 4 email sequence that I can send to industrial manufacturing companies about the benefits of VNTANA', then you should generate 'cold email' as one of the concepts. These concepts should be separated by commas.'"""},
                {"role": "user", "content": "Please generate your semantic search query based on the user request:"},
                {"role": "assistant", "content": input}
            ]
        )
        weaviate_query = response['choices'][0]['message']['content']
        logging.info("Search query generated successfully.")  # Changed from print to logging.info
        return weaviate_query
    except Exception as e:
        logging.error(f"Error generating query with OpenAI: {e}")  # Changed from print to logging.error
        return None

search = SerpAPIWrapper()
vntana = VNTANAsalesQueryTool()

# Load tools and memory
math_llm = OpenAI(temperature=0.0, model="gpt-4", streaming=True)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

additional_tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
    Tool(
        name = "VNTANA Sales and Marketing Tool",
        func=vntana.run,  # Use run here instead of arun
        description="useful when you need to write any information about VNTANA or sales and marketing copy"
    )
]

tools.extend(additional_tools) # Add the additional tools to the original list

llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo-16k", streaming=False)

memory = ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True, max_tokens=4200, llm=llm)

# Create the agent and run it
st_container = st.container()
llm = ChatOpenAI(
    temperature=0.4, 
    callbacks=[StreamlitCallbackHandler(parent_container=st_container, expand_new_thoughts=False, collapse_completed_thoughts=True)], 
    streaming=True,
    model="gpt-4",
)

# Create the agent
agent = CustomChatAgent.from_llm_and_tools(llm, tools, output_parser=CustomOutputParser(), handle_parsing_errors=True)
chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory, stop=["Observe:"])

# Initialize the chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit interaction
st.title("VNTANA Customer Support Assistant")

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

def convert_to_json(data):
    """Convert a string to a JSON object. If the string is not in valid JSON format, 
    wrap it in a dictionary with the key 'AI'. If the data is already a dictionary, return it as-is."""
    if isinstance(data, dict):
        return data

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        # If not a valid JSON string, wrap it with the key 'AI'
        return {"AI": data}


def parse_ai_response(response_data):
    logging.info("Response Data: %s", response_data)
    
    # If response_data is a string, try to parse it as JSON
    if isinstance(response_data, str):
        try:
            response_dict = json.loads(response_data)
        except json.JSONDecodeError:
            # If the string is not a valid JSON, return the original response_data
            return response_data
    elif isinstance(response_data, dict):
        response_dict = response_data
    else:
        return "Invalid input type"

    # Check if the response contains the "Non-JSON Response" key
    if "Non-JSON Response" in response_dict:
        return response_dict["Non-JSON Response"]

    # Extract the AI's response from the response dictionary
    ai_response = response_dict.get("AI", "")
        
    # If the AI's response is not found, extract the text between "AI": " and "
    if not ai_response and isinstance(response_dict, str):
        match = re.search(r'"AI": "(.*?)"', response_dict)
        if match:
            ai_response = match.group(1)

    # Extract the actual response after "Observation: "
    observation_index = ai_response.find("Observation: ")
    if observation_index != -1:
        ai_response = ai_response[observation_index + len("Observation: "):]
    else:
        ai_response = "Observation not found in response."

    # Remove any leading or trailing whitespace
    ai_response = ai_response.strip()

    return ai_response


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

if prompt := st.chat_input():
    st.chat_message("Gardener", avatar="👨‍🌾🌱" ).write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})  # Add user message to chat history
    with st.chat_message("AI", avatar = "🤖🥬"):
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









