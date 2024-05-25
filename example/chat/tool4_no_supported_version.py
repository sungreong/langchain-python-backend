from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from operator import itemgetter
from langchain_core.runnables import (
    Runnable,
)
from langchain_experimental.tools.python.tool import PythonREPLTool, PythonInputs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


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
    # llm_with_tools = llm.bind_tools(tools)
    from langchain.tools.render import render_text_description

    # rendered_tools = render_text_description(tools)

    rendered_tools = "\n".join([str(format_tool_to_openai_function(t)) for t in tools])

    # llm.stream_runnable = False

    system_prompt = f"""
    You are an assistant that has access to the following set of tools. 
    Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys.
    To use these functions respond with:\n {{"name": "function_name", "arguments":  {{"arg_1": "value_1", "arg_1": "value_1", ... }}}} \n

    This version of the system prompt underscores the importance of focusing on JSON creation and minimizes any other text or guidance.
    """
    # ("user", system_prompt),

    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])

    chain = prompt | llm | JsonOutputParser()

    # query = "Using Python as a tool, write a function to calculate Fibonacci numbers. Ensure the function is correctly implemented to avoid syntax errors, and then execute it to find the 10th Fibonacci number."
    # print(chain.invoke(dict(input=query)))
    # print(chain.invoke({"input": "what's thirteen times 4"}))
    def tool_chain(model_output):
        print(model_output)
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool

    chain = chain | tool_chain
    # query = "Using Python as a tool, write a function to calculate Fibonacci numbers. Ensure the function is correctly implemented to avoid syntax errors, and then execute it to find the 100th Fibonacci number."

    # print(chain.invoke(dict(input=system_prompt + query)))

    query = """
    Using Python as a tool, 
    write a function to pandas check type and change int to string using `df`. 
    Ensure the function is correctly implemented to avoid syntax errors, 
    """

    print(chain.invoke(dict(input=system_prompt + query)))
    print(chain.invoke({"input": system_prompt + "what's 3 plus 1132"}))


# https://python.langchain.com/v0.1/docs/use_cases/tool_use/prompting/
# https://python.langchain.com/v0.1/docs/use_cases/tool_use/multiple_tools/
