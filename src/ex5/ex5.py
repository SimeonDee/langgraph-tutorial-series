"""
Demonstrate How to Implement Simple Looping Graph with LangGraph

Main Goal:
- Learn how to code 'Looping Logic'.

Objectives:
- Implement 'looping logic' to route the flow of data back to the node.
- Create a 'conditional edge' to handle decision making and control graph flow.

Problem Description:
You are tasked with implementing a graph with 'loop'.

Requirements:
- input:
initial_state = AgentState(username=10, counter=5, random_size=5, \
    random_numbers=[], message="")
"""

import os
import random
from langgraph.graph import StateGraph, START, END
from pprint import PrettyPrinter
from typing import List, TypedDict


printer = PrettyPrinter(indent=3, sort_dicts=False)


###########
# State
###########


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    username: str  # User's name
    counter: int  # Counter for the loop
    random_size: int  # Size of the random numbers list
    random_numbers: List[int]  # List to store random numbers
    message: str  # Message to display to the user


############
# Nodes
############


def greet_user_node(state: AgentState) -> AgentState:
    """
    Greet the user.
    """
    state["message"] += f"Hello, {state['username']}!\n"
    return state


def random_number_generator_node(state: AgentState) -> AgentState:
    """
    Generate a random number and add it to the list.
    """
    random_number = random.randint(1, 100)
    state["random_numbers"].append(random_number)
    state["message"] += f"Generated number: {random_number}\n"
    state["counter"] += 1

    return state


def farewell_user_node(state: AgentState) -> AgentState:
    """
    Farewell the user and display the random numbers.
    """
    state["message"] += "Goodbye!\n"
    state[
        "message"
    ] += f"\nRandom numbers generated: {state['random_numbers']}\n"  # noqa: E501
    return state


def loop_condition_node(state: AgentState) -> str:
    """
    Check if the loop should continue.
    """
    if state["counter"] < state["random_size"]:
        state["message"] += "Continuing the loop...\n"
        return "loop"
    else:
        state["message"] += "Exiting the loop.\n"
        return "exit"  # End the loop


###########
# Graph
###########


graph = StateGraph(state_schema=AgentState)

# Add nodes to the graph
graph.add_node("greet_user", greet_user_node)
graph.add_node("random_number_generator", random_number_generator_node)
graph.add_node("farewell_user", farewell_user_node)

# Add edges with conditional logic
graph.add_edge(START, "greet_user")
graph.add_edge("greet_user", "random_number_generator")
graph.add_conditional_edges(
    "random_number_generator",
    loop_condition_node,  # The loop condition decider
    {"loop": "random_number_generator", "exit": "farewell_user"},
)

# Add the farewell node to the end of the graph
graph.add_edge("farewell_user", END)

app = graph.compile()


###############
# Main function to run the graph
###############


def main(app: StateGraph, initial_state: AgentState) -> None:
    """
    Run the graph with the initial state.
    """
    # Print the initial state
    print("")
    print("Initial State:")
    printer.pprint(initial_state)
    print("\n---\n")

    # Invoke the graph
    final_state = app.invoke(initial_state)

    # Print the final state
    print("Final State:")
    printer.pprint(final_state)
    print("\n---\n")

    print("Message:")
    print(final_state["message"])
    print("\n---\n")


if __name__ == "__main__":
    # Setting up initial state
    initial_state = AgentState(
        username="Adewale",
        counter=0,
        random_size=8,
        random_numbers=[],
        message="",
    )

    # Run the main function
    main(app, initial_state)

    # Save the graph structure
    outpute_path = os.path.join("src", "ex5", "graph_structure_ex5.png")
    app.get_graph().draw_mermaid_png(output_file_path=outpute_path)
    print(f"Graph Structure saved as {outpute_path}")
