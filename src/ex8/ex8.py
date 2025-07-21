"""
Simple AI Agents using LangGraph (with a simple Memory and Window size)
=======================================================================

Goal: To learn how to build a simple AI agent using LangGraph.

Objectives:
- Implement a simple AI agent that can respond to user input.
- Use LangGraph to manage the state and flow of the agent.
- Integrate a memory mechanism to keep track of the conversation history.
"""

import os
from datetime import datetime
from pprint import PrettyPrinter
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    # ToolMessage,
)
from dotenv import load_dotenv

load_dotenv()
# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

LOG_FILE_PATH = os.path.join("src", "ex8", "conversation_logs.log")
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

    messages: List[dict]  # List of messages in the conversation


#############
# Nodes
#############


def process_query_node(state: AgentState) -> AgentState:
    """Generates a response from the AI based on the user's input query.
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


#############
# The Graph
#############


graph = StateGraph(state_schema=AgentState)

# add nodes
graph.add_node("query_processor", process_query_node)

# Add edges
graph.add_edge(START, "query_processor")
graph.add_edge("query_processor", END)

agent = graph.compile()


###########################
# Save the graph structure
###########################


output_path = os.path.join("src", "ex8", "graph_structure_ex8.png")
agent.get_graph().draw_mermaid_png(output_file_path=output_path)
print(f"Graph Structure saved as {output_path}")
print("\n---\n")


################
# Execute Graph
################


def log_conversation_histories(histories: list) -> None:
    """Logs all conversation history to file."""
    current_date_time = datetime.strftime(
        datetime.now(),
        format="%d-%m-%Y, %H:%M:%S",
    )
    with open(LOG_FILE_PATH, "a") as file:
        file.write(f"\nDate: {current_date_time}\n")
        for message in histories:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n\n")
    print(f"Conversation history saved to {LOG_FILE_PATH}.")


def main(
    agent: StateGraph,
    initial_state: AgentState,
    window_size: int = 4,
) -> None:
    """Executes the agent graph."""

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
            conversation_history.append(
                AIMessage(
                    content="Exiting the chat. Goodbye!",
                )
            )
            print("AI: Exiting the chat. Goodbye!")
            break

        # execute agent graph
        response = agent.invoke({"messages": conversation_history})
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

    # Log conversation history
    log_conversation_histories(conversation_history)


if __name__ == "__main__":
    # Intializing the state
    system_prompt = SystemMessage(
        content=(
            "You are a helful AI Assistant. Answer "
            "user queries to the best of your ability."
        ),
    )
    initial_state = AgentState(
        messages=[system_prompt],
    )
    main(
        agent,
        initial_state,
        window_size=4,  # Adjust the window size as needed, for memory mangt
    )
