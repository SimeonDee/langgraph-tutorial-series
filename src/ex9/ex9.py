"""
Implementing Simple ReAct Agents (Agent with Tools) using LangGraph
====================================================================

Goal: To learn how to build a Simple ReAct agent using LangGraph.

Objectives:
- Learn how to create "Tools" in LangGraph.
- How to create a "ReAct Graph".
- Work with different types of "Messages" such as "ToolMessage"
- Integrate tool-use into Agents with LangGraph.
- Test robustness of our graph.

# Sample Query:
"Add 24 to 76. Then divide the result by 20. Multiply the answer by 2 and
finally subract 4 from the computed answer."
"""

import os
from pprint import PrettyPrinter
from typing import Annotated, Any, Iterable, Sequence, TypedDict, Union

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    # ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


from dotenv import load_dotenv

load_dotenv()
# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

printer = PrettyPrinter(indent=3, sort_dicts=False)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_retries=3,
    max_tokens=1000,
    request_timeout=60,
    verbose=True,
)

###########
# State
###########


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    messages: Annotated[
        Sequence[BaseMessage], add_messages
    ]  # Sequence of messages in the conversation


#############
# Tools
#############


@tool
def add(a: int, b: int) -> int:
    """A function that adds two numbers and return their result.

    Args:
        a (int): the first value.
        b (int): the second value.

    Returns:
        int: The addition of a and b.
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """A function that performs subtraction of numbers and
    return their result.

    Args:
        a (int): the first value.
        b (int): the second value.

    Returns:
        int: The result.
    """
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """A function that multiplies or find the product of two
    numbers and return their result.

    Args:
        a (int): the first value.
        b (int): the second value.

    Returns:
        int: The multiplication/product of a and b.
    """
    return a * b


@tool
def divide(a: int, b: int) -> Union[int, float]:
    """A function that divides two numbers and return their result.

    Args:
        a (int): the first value.
        b (int): the second value.

    Returns:
        float: The result of dividing a by b.
        str: Error message
    """
    try:
        if b == 0:
            raise Exception("You cannot divide by 0")
        return a / b
    except Exception as e:
        return f"{e}"


tools = [add, subtract, multiply, divide]

# Bind llm to tools
llm = llm.bind_tools(tools)

#############
# Nodes
#############


def display_tool_call_report(tool_call: dict) -> None:
    """Utility function to display tool called."""
    if tool_call["type"] == "function":
        func_name = tool_call["function"]["name"]
        args = eval(tool_call["function"]["arguments"])
        args_str = ",".join([f"{k}={v}" for k, v in args.items()])
        print(f"\tCalling tool -- {func_name}({args_str})")


def model_call(state: AgentState) -> AgentState:
    """Generates a response from the AI based on the user's input query.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        AgentState: The updated state with the AI's response.
    """
    try:
        response = llm.invoke(state["messages"])
        # print if response has content (i.e. not a tool call)
        if response.content:
            print(f"AI: {response.content}\n")
        elif response.additional_kwargs.get("tool_calls", ""):
            # Report the tool calls
            for tool_call in response.additional_kwargs["tool_calls"]:
                display_tool_call_report(tool_call)
        return {"messages": [response]}
    except Exception as e:
        print(f"AI: Error occured {e}\n")
        return state


def should_continue(state: AgentState) -> str:
    """Decides whether the agent should conintue or end."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


#############
# The Graph
#############


graph = StateGraph(state_schema=AgentState)


# add agent nodes
graph.add_node("agent", model_call)

# add tools
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "agent")


app = graph.compile()


###########################
# Save the graph structure
###########################


output_path = os.path.join("src", "ex9", "graph_structure_ex9.png")
app.get_graph().draw_mermaid_png(output_file_path=output_path)
print(f"Graph Structure saved as {output_path}")
print("\n---\n")


################
# Execute Graph
################


# A stream printer utility func
def print_stream(stream_data: Iterable[Any]):
    for data in stream_data:
        message = data["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            printer.pprint(message)


# main func
def main(
    app: StateGraph,
    initial_state: AgentState,
    window_size: int = 4,
) -> None:
    """Executes the app graph."""

    print("\nInitial State:")
    printer.pprint(initial_state)

    # Initialize conversation history
    conversation_history = initial_state["messages"]

    # A turn is a pair of user input and AI response + the system message
    conversation_turns = window_size * 2 + 1

    while True:
        # Get user input
        user_input = input("\nYou: ")

        if not user_input.strip():
            print("AI: Please enter a valid query.")
            continue

        # Append user input to history
        conversation_history.append(HumanMessage(content=user_input))

        if user_input.lower() in ["exit", "quit", "bye", "goodbye", "q"]:
            print("AI: Exiting the chat. Goodbye!")
            break

        # execute agent graph
        response = app.invoke({"messages": conversation_history})
        conversation_history = response["messages"]

        # Limit the conversation history to the last `window_size`
        # conversation turns
        # This is to prevent the model from being overwhelmed with too
        # much context
        # This is a simple memory mechanism
        # If the conversation history exceeds the window size, truncate it
        if len(conversation_history) > conversation_turns:
            conversation_history = conversation_history[-conversation_turns:]

    print("\nFinal State:")
    printer.pprint({"messages": conversation_history})


if __name__ == "__main__":
    # Intializing the state
    # system_prompt = SystemMessage(
    #     content=(
    #         "You are a helful AI Assistant. Answer "
    #         "user queries to the best of your ability "
    #         "using only available tools. "
    #         "Do not guess an answer, say 'you don't know the answer' "
    #         "if there is no available tool to answer the question."
    #     ),
    # )
    system_prompt = SystemMessage(
        content=(
            "You are a helful AI Assistant. Answer "
            "user queries to the best of your ability using available "
            "tools first."
        ),
    )
    initial_state = AgentState(
        messages=[system_prompt],
    )
    main(
        app,
        initial_state,
        window_size=4,  # Adjust the window size as needed, for memory mangt
    )
