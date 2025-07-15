import os
from typing import List, TypedDict, Union
from langgraph.graph import StateGraph
from IPython.display import display, Image

"""
Demonstrate Multiple Inputs Graph

Objectives: Learn how to handle multiple inputs
- Define a more complex AgentState
- Create a processing node that performs operations on list data
- Setup LangGraph that processes and outputs computed results
- Invoke the graph with structured inputs and retrieve outputs
"""

"""
Problem Statement:
    You are tasked with creating a LangGraph application that processes \
    a list of numbers.

Requirements:
- The application should accept a list of integers, a user name, \
    and an operation type.
- Operation type can be one of 'sum', 'product', 'min', 'max', 'avg'.
- It should compute the appropriate operation on the numbers in the list.
- The application should return the results in a structured format.
"""

#################################################
# Agent State
#################################################


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    numbers: List[int]  # List of integers to process
    user_name: str  # Name of the user
    operation: str  # Operation type ('sum', 'product', 'min', 'max', 'avg')
    result: Union[int, float]  # Result of the operation
    message: str  # Message to display to the user


#################################################
# Nodes
#################################################


def process_numbers_node(state: AgentState) -> AgentState:
    """
    A node that processes the list of numbers based on the operation type.
    """
    if state["operation"].lower() == "sum":
        state["result"] = sum(state["numbers"])
    elif state["operation"].lower() == "product":
        result = 1
        for num in state["numbers"]:
            result *= num
        state["result"] = result
    elif state["operation"].lower() == "min":
        state["result"] = min(state["numbers"])
    elif state["operation"].lower() == "max":
        state["result"] = max(state["numbers"])
    elif state["operation"].lower() == "avg":
        state["result"] = (
            sum(state["numbers"]) / len(state["numbers"])
            if state["numbers"]
            else 0  # noqa: E501
        )
    else:
        raise ValueError("Invalid operation type provided.")

    state["message"] = (
        f"Hello {state['user_name']}, the result of your requested "
        f"'{state['operation']}' operation is {state['result']}."
    )
    return state


def farewell_node(state: AgentState) -> AgentState:
    """
    A node that bids farewell to the user.
    """
    state["message"] = (
        f"{state['message']} \n\n---\n\nGoodbye {state['user_name']}!"  # noqa: E501
    )
    return state


#################################################
# Graph
#################################################

graph = StateGraph(state_schema=AgentState)

# Adding nodes to the graph
graph.add_node("process_numbers", process_numbers_node)
graph.add_node("farewell", farewell_node)

# Adding edges to connect the nodes
graph.add_edge(start_key="process_numbers", end_key="farewell")

# Setting the entry and finish points of the graph
graph.set_entry_point("process_numbers")
graph.set_finish_point("farewell")

# Compile the graph into an application
app = graph.compile()

##############################################
# Executing the Graph (application)
##############################################
"""
- Setting up the graph with a starting state
"""

initial_state = AgentState(
    numbers=[1, 2, 3, 4, 5],
    user_name="Sanjo",
    operation="sum",
    result=0,
    message="",
)

# Executing the application
result = app.invoke(initial_state)

# Alternatively, you can invoke the application with a different
# initial state as a dict
# Uncomment the line below to test with a different operation
# result = app.invoke(
#     {
#         "numbers": [10, 20, 30],
#         "user_name": "Ade",
#         "operation": "product",
#         "result": 0,
#         "message": "",
#     }
# )  # noqa: E501


##############################################
# Displaying the result
##############################################

print("Final Result:")
print(f"Numbers: {result['numbers']}")
print(f"User Name: {result['user_name']}")
print(f"Operation: {result['operation']}")
print(f"Result: {result['result']}")
print(f"Message: {result['message']}")
print("\n---\n")

# Displaying the final result
print("Final Agent State:")
print(result)
print("\n---\n")


##############################################
# Displaying and Saving the graph structure
##############################################
"""
NOTE:
- Internet connection is required to display or save the graph.
- The graph structure can be visualized using the draw_mermaid_png method.
- This will generate a PNG image of the graph structure.
"""

# Displaying the graph as an image (will only disolay in Jupyter Notebook)
graph_image = app.get_graph().draw_mermaid_png()
display(Image(graph_image))

# Save the graph structure
outpute_path = os.path.join("src", "ex2", "graph_structure_ex2.png")
app.get_graph().draw_mermaid_png(output_file_path=outpute_path)
print(f"Graph Structure saved as {outpute_path}")
