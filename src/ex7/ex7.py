"""
AI Agents using LangGraph
================================
"""

import os
from pprint import PrettyPrinter
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI

# from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    # ToolMessage,
)

from dotenv import load_dotenv

load_dotenv()
# Set environment variables
# os.environ["LANGCHAIN_HANDLER"] = "langgraph"
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

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful AI assistant."),
#         ("user", "{user_input}"),
#         ("assistant", "{response}"),
#     ]
# )


###########
# State
###########


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    username: str  # User's name
    messages: List[dict]  # List of messages in the conversation


#############
# Nodes
#############


def get_user_input_node(state: AgentState) -> AgentState:
    """Prompts user for input and updates the state with the message.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        AgentState: The updated state with the user's message.
    """
    user_input = input("\nYou: ")
    state["messages"].append(HumanMessage(content=user_input))
    return state


def get_response_node(state: AgentState) -> AgentState:
    """Generates a response from the AI based on the user's input.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        AgentState: The updated state with the AI's response.
    """
    try:
        response = llm.invoke(state["messages"])
        print(f"AI: {response.content}\n")
        state["messages"].append(AIMessage(content=response.content))
    except Exception as e:
        print(f"AI: Error occured {e}\n")
    finally:
        return state


def loop_decider_router(state: AgentState) -> str:
    """Determines the next node based on the last human message in the state.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        str: The string to determine the next node to transition to.
            e.g. "input", "continue", or "exit".
    """
    if len(state["messages"]) == 0:
        return "input"
    # Check if the last message is a human message
    if not state["messages"] or not isinstance(state["messages"][-1], dict):
        return "input"
    if not isinstance(state["messages"][-1], HumanMessage):
        return "input"
    # Check if the last message is a farewell or exit command
    # This is a simple check, you can expand it to include more variations
    if state["messages"][-1].content.lower() in [
        "quit",
        "exit",
        "q",
        "bye",
        "goodbye",
        "stop",
        "end",
    ]:
        return "exit"
    else:
        return "continue"


def farewell_node(state: AgentState) -> AgentState:
    """Handles the farewell message and ends the conversation.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        AgentState: The state after the farewell message.
    """
    print(f"AI: Goodbye, {state["username"]}!\n\n")
    return state


#############
# The Graph
#############


graph = StateGraph(state_schema=AgentState)

# add nodes
graph.add_node("query_getter", get_user_input_node)
graph.add_node("response_getter", get_response_node)
graph.add_node("farewell", farewell_node)

# Add edges
graph.add_edge(START, "query_getter")
graph.add_conditional_edges(
    "query_getter",
    loop_decider_router,
    {
        "input": "query_getter",
        "continue": "response_getter",
        "exit": "farewell",
    },
)
graph.add_edge("response_getter", "query_getter")
graph.add_edge("farewell", END)

agent = graph.compile()


################
# Execute Graph
################


def main(agent: StateGraph, initial_state: AgentState) -> None:
    """Executes the agent graph."""

    print("\nInitial State:")
    printer.pprint(initial_state)

    # execute agent graph
    result = agent.invoke(initial_state)

    print("\nFinal State:")
    printer.pprint(result)


if __name__ == "__main__":
    # Intializing the state
    initial_state = AgentState(
        username="Kunle",
        messages=[SystemMessage(content="You are a helful AI Assistant.")],
    )
    main(agent, initial_state)
