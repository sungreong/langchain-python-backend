from langchain_core.tools import BaseTool

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


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


from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnablePassthrough

from typing import List, Union
from langchain.agents import AgentExecutor, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
import re
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        if "Final Answer:" in llm_output:
            # TODO: 해당 로직은 제대로 작동하지 않음
            # PROMPT에 따라서 해당 결과가 나올 지 안 나올 지 모르기 때문에
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"<functioncall>(.*?)</functioncall>"
        match = re.search(regex, llm_output, re.DOTALL)
        import json

        print(f"{llm_output=}")
        try:
            import ast

            action_input = match.group(1).strip()
            data_dict = ast.literal_eval(action_input)
            json_string = json.dumps(data_dict)
            parsed_json = json.loads(json_string)
            action = parsed_json["name"]  # JSON 구조 내의 'name' 키에서 action을 추출
            tool_input = parsed_json["arguments"]  # 'arguments' 키에서 tool_input을 추출
        except Exception as e:
            print("ERROR", e, llm_output)
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        else:
            llm_output = f"<functioncall>{parsed_json}</functioncall>"
            return AgentAction(tool=action, tool_input=tool_input, log=llm_output)


if __name__ == "__main__":
    from langchain import hub

    toolkit = [add, multiply, square]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    llm = ChatOpenAI(
        model_name="microsoft/Phi-3-mini-4k-instruct",  # Ensure this matches a model your server can serve
        openai_api_base="http://localhost:8000/v1",  # Change this to your local server's URL
        openai_api_key="Not needed for local server",  # API key not required for local setups
        openai_proxy="",  # Endpoint for generating responses
        temperature=0.0,  # Deterministic output,
        max_tokens=3000,  # Maximum tokens to generate
        stream=False,
        # request_timeout=120,  # Timeout for requests
        frequency_penalty=0.0,
        top_p=1.0,
        streaming=True,
    )
    print(prompt)
    # TODO: `create_openai_tools_agent` is not not working
    # agent = create_openai_tools_agent(llm, toolkit, prompt)
    ##############
    functions = [format_tool_to_openai_function(t) for t in toolkit]
    llm_with_tools = llm.bind_functions(functions=functions)
    agent = (
        RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"]))
        | prompt
        | llm_with_tools
        | CustomOutputParser()
    )
    ##############
    agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
    result = agent_executor.invoke({"input": "hi"})
    print(result)

    result = agent_executor.invoke(
        {
            "input": "what's 3 minus 1132",
            "chat_history": [
                HumanMessage(content="what's 3 times 1132"),
                AIMessage(content="15"),
            ],
        }
    )
    print(result)
