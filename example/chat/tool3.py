from langchain.tools.render import format_tool_to_openai_function

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.runnables import Runnable


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


from langchain_core.messages import AIMessage
from langchain_experimental.tools.python.tool import PythonREPLTool, PythonInputs

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

    functions = [format_tool_to_openai_function(t) for t in tools]
    # llm.stream_runnable = False

    llm_with_tools = llm.bind_functions(functions=functions)

    tool_map = {tool.name: tool for tool in tools}

    def call_tools(msg: AIMessage) -> Runnable:
        """Simple sequential tool calling helper."""
        print(msg)
        tool_map = {tool.name: tool for tool in tools}
        tool_calls = msg.tool_calls.copy()
        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        return tool_calls

    chain = llm_with_tools | call_tools
    query = "Using Python as a tool, write a function to calculate Fibonacci numbers. Ensure the function is correctly implemented to avoid syntax errors, and then execute it to find the 10th Fibonacci number."
    print(chain.invoke(query))
    print(chain.invoke("What's 23 times 7"))
# https://python.langchain.com/v0.1/docs/use_cases/tool_use/multiple_tools/
