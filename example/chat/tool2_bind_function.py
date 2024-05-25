from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
import re
from langchain_experimental.tools.python.tool import PythonREPLTool, PythonInputs


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


# https://pub.aimind.so/building-a-custom-chat-agent-for-document-qa-with-langchain-gpt-3-5-and-pinecone-e3ae7f74e5e8

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
    tools = [add, multiply, square, subtract, PythonREPLTool(args_schema=PythonInputs)]
    prompt = ChatPromptTemplate.from_messages(
        [
            # (
            #     "system",
            #     system_msg
            # ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    chat_history = []
    functions = [format_tool_to_openai_function(t) for t in tools]
    llm_with_tools = llm.bind_functions(functions=functions)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | CustomOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        early_stopping_method="generate",
    ).with_config({"run_name": "Agent"})
    query = "Using Python as a tool, write a function to calculate Fibonacci numbers. Ensure the function is correctly implemented to avoid syntax errors, and then execute it to find the 10th Fibonacci number."
    result = agent_executor.invoke({"input": query, "chat_history": []})
    ai_content = result["intermediate_steps"][0][0].log if len(result["intermediate_steps"]) > 0 else result["output"]
    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=ai_content),
        ]
    )
    query = "What is (10.123123123 - 5.9882) ?"
    result = agent_executor.invoke({"input": query, "chat_history": chat_history})
    ai_content = result["intermediate_steps"][0][0].log if len(result["intermediate_steps"]) > 0 else result["output"]
    chat_history.extend(
        [
            HumanMessage(content=query),
            AIMessage(content=ai_content),
        ]
    )
    query = "What is (10.123123123 ** 2) ?"
    result = agent_executor.invoke({"input": query, "chat_history": chat_history})
