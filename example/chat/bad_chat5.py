from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
from typing import Optional, Type
from langchain.tools import BaseTool, StructuredTool, tool


from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents.format_scratchpad import format_to_openai_function_messages

from langchain_openai import ChatOpenAI, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together"""  # this docstring gets used as the description
    return a + b  # the actions our tool performs


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
import re


from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.output_parsers.json import parse_json_markdown


class CustomOutputParser(AgentOutputParser):

    def parse(self, text: str) -> any:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(text)
            print(f"{response=}")
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(tool=action, tool_input=action_input, log=text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish(return_values={"output": text}, log=text)


from typing import List
from langchain.agents import Tool
from langchain.schema import SystemMessage
from langchain.prompts import BaseChatPromptTemplate


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    prefix: str
    instructions: str
    sufix: str
    # The list of tools available
    tools: List[Tool]

    def _set_tool_description(self, tool_description, tool_name, tool_input):
        full_description = f"""{tool_description}, send this:
```json
{{"action": "{tool_name}",
"action_input": "{tool_input}"}}
```
"""
        return full_description

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        separator = ". Input:"

        kwargs["tools"] = "\n".join(
            [f"{self._set_tool_description(tool.description, tool.name, tool.description)}" for tool in self.tools]
        )

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        # Format the instructions replacing the variables with the values
        formatted = self.instructions.format(**kwargs)

        # Add the sufix
        formatted += self.sufix
        return [SystemMessage(content=formatted)]


TEMPLATE_INSTRUCTIONS = """You are an Assistant designed to assist with a 
wide range of tasks.

Choose one of the following tools to use based on the user input:

- {tools}

- When you need to respond to other user's utterances or generate the 
response from the other tools, send this:
    ```json
    {{"action": "Final Answer",
      "action_input": "the final answer to the original input question"}}
    ```
- When you need to generate the response from the other tools, send this:
    ```json
    {{"action": "Final Answer",
      "action_input": "the response from the tool in a prhase"}}
    ```
Current conversation:
{history}
<|user|>
{input}
<|assistant|>
{agent_scratchpad}"""

SUFFIX = """\nRespond ONLY in JSON format!<|end|>"""

if __name__ == "__main__":
    llm = ChatOpenAI(
        model_name="microsoft/Phi-3-mini-4k-instruct",  # Ensure this matches a model your server can serve
        openai_api_base="http://localhost:8000/v1",  # Change this to your local server's URL
        openai_api_key="Not needed for local server",  # API key not required for local setups
        openai_proxy="",  # Endpoint for generating responses
        temperature=0.0,  # Deterministic output,
        max_tokens=3000,  # Maximum tokens to generate
        stream=True,
        timeout=180,  # Timeout for requests
        frequency_penalty=0.0,
        top_p=1.0,
        streaming=True,
    )

    # tools = [CustomCalculatorTool()]

    tools = [add, multiply, square, subtract]
    functions = [format_tool_to_openai_function(t) for t in tools]
    print(functions)
    # llm_with_tools = llm.bind_functions(functions=functions)
    prompt = CustomPromptTemplate(
        prefix="",
        instructions=TEMPLATE_INSTRUCTIONS,
        sufix=SUFFIX,
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"],
    )
    from langchain.memory import ConversationBufferWindowMemory

    memory = ConversationBufferWindowMemory(memory_key="history", k=5, return_messages=True)
    from langchain.agents import LLMSingleActionAgent, AgentExecutor, Tool

    from langchain.agents import AgentExecutor
    from langchain import LLMChain
    from langchain.agents import LLMSingleActionAgent, AgentExecutor, Tool

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool["name"] for tool in functions]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=CustomOutputParser(),
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, max_iterations=3, memory=memory
    )  # , "stream_runnable": False
    print("result", "-" * 20)
    query = "What is (3111231 * 121231) ?"
    result = agent_executor.run({"input": query, "history": []})
    # print(result["intermediate_steps"][0][0].log)
    ai_content = result["intermediate_steps"][0][0].log if len(result["intermediate_steps"]) > 0 else result["output"]
    query = "What is (10.123123123 - 5.9882) ?"

    result = agent_executor.invoke({"input": query})
    print("result", "-" * 20)
    print(result)
    ai_content = result["intermediate_steps"][0][0].log if len(result["intermediate_steps"]) > 0 else result["output"]
    print(ai_content)
    print("-" * 20)
    query = "What is (10.123123123 ** 2) ?"
    result = agent_executor.invoke({"input": query})
    print("result", "-" * 20)
    print(result)
    print("-" * 20)
# https://pub.aimind.so/building-a-custom-chat-agent-for-document-qa-with-langchain-gpt-3-5-and-pinecone-e3ae7f74e5e8
