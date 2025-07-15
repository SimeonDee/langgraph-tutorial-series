"""
Demonstration of a simple state graph using LangGraph.

- This example showcases how to create a state graph with nodes that greet \
    and bid farewell to the user.
- It includes the definition of the state, nodes, and the graph structure.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph
from IPython.display import display, Image


#############################################
# Agent State
#############################################
"""
- The AgentState class defines the structure of the state used in the graph.
- It is a TypedDict, which allows for type checking and autocompletion in IDEs.
- The state contains a single key 'message' which is a string.
"""


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    message: str


#############################################
# Nodes
#############################################
"""
- These are functions that will be executed in the graph.
- They take the current state as input and return the updated state.
"""


def greeting_node(state: AgentState) -> AgentState:
    """
    A node that greets the user.
    """
    state["message"] = (
        f"{state['message']}, You are doing an amazing job learning LangGraph."
    )
    return state


def farewell_node(state: AgentState) -> AgentState:
    """
    A node that bids farewell to the user.
    """
    state["message"] = f"{state['message']} Goodbye!"
    return state


#############################################
# Graph
#############################################
"""
- Create a state graph with the defined state schema
"""

graph = StateGraph(state_schema=AgentState)

# adding nodes to the graph
graph.add_node("greeter", greeting_node)
graph.add_node("farewell", farewell_node)
graph.add_edge(start_key="greeter", end_key="farewell")

# setting the entry and finish points of the graph
graph.set_entry_point("greeter")
graph.set_finish_point("farewell")

# Compile the graph into an application
app = graph.compile()

##############################################
# Displaying and Saving the graph structure
##############################################
"""
NOTE:
- Internet connection is required to display or save the graph.
- The graph structure can be visualized using the draw_mermaid_png method.
- This will generate a PNG image of the graph structure.
"""

# Displaying the graph as an image
graph_image = app.get_graph().draw_mermaid_png()
display(Image(graph_image))

# Save the graph structure
outpute_path = os.path.join("src", "ex1", "graph_structure_ex1.png")
app.get_graph().draw_mermaid_png(output_file_path=outpute_path)
print(f"Graph Structure saved as {outpute_path}")


##############################################
# Main
##############################################


def main(app: StateGraph):
    # Setting up initial state of the application
    initial_state = AgentState(message="Sanjo")

    # Invoking the application with the initial state
    result = app.invoke(initial_state)

    # Alternatively, you can invoke the application with a different
    # initial state as a dict
    # Uncomment the line below to test with a different message
    # result = app.invoke({"message": "Ade"})

    print("\n---\n")
    print(result)
    print("\n---\n")


if __name__ == "__main__":
    main(app)
