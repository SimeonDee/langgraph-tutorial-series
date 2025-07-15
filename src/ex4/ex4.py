import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pprint import PrettyPrinter

"""
Demonstrate How to Implement Simple Conditional Graph in LangGraph

Main Goal:
- Learn how to use "conditional_edges".

Objectives:
- Implement 'conditional logic' to route the flow of data to different nodes.
- Use 'START' and 'END' nodes to manage entry and exit points explicitly.
- Design multiple nodes to perform different arithmetic operations: \
 (addition (+), substraction (-), multiplication (*), division (/))
- Create a 'route node' to handle decision making and control graph flow.
"""

"""
Problem Description:
You are tasked with implementing a graph with 2 'conditional edges'.

Requirements:
- input:
initial_state = AgentState(val1=10, operation='+', val2=5, val3=7, \
    operation2='-', val4=2, result1=0, result2=0)
"""


printer = PrettyPrinter(indent=3, sort_dicts=False)


###########
# State
###########


class AgentState(TypedDict):
    val1: int
    val2: int
    val3: int
    val4: int
    operation1: str
    operation2: str
    result1: int
    result2: int
    message: str


###########
# Nodes
###########


def adder1_node(state: AgentState) -> AgentState:
    """Add val1 and val2."""
    state["result1"] = state["val1"] + state["val2"]
    state["message"] += (
        f"Addition performed on {state['val1']} and "
        f"{state['val2']}. Result: {state['result1']}"
    )  # noqa: E501
    return state


def adder2_node(state: AgentState) -> AgentState:
    """Add val3 and val4."""
    state["result2"] = state["val3"] + state["val4"]
    state["message"] += (
        f"\nAddition performed on {state['val3']} and "
        f"{state['val4']}. Result: {state['result2']}"
    )  # noqa: E501
    return state


def subtractor1_node(state: AgentState) -> AgentState:
    """Subtract val1 and val2."""
    state["result1"] = state["val1"] - state["val2"]
    state["message"] += (
        f"Subtraction performed on {state['val1']} and "
        f"{state['val2']}. Result: {state['result1']}"
    )  # noqa: E501
    return state


def subtractor2_node(state: AgentState) -> AgentState:
    """Subtract val3 and val4."""
    state["result2"] = state["val3"] - state["val4"]
    state["message"] += (
        f"\nSubtraction performed on {state['val3']} and "
        f"{state['val4']}. Result: {state['result2']}"
    )  # noqa: E501
    return state


def multiplier1_node(state: AgentState) -> AgentState:
    """Multiply val1 and val2."""
    state["result1"] = state["val1"] * state["val2"]
    state["message"] += (
        f"Multiplication performed on {state['val1']} and "
        f"{state['val2']}. Result: {state['result1']}"
    )  # noqa: E501
    return state


def multiplier2_node(state: AgentState) -> AgentState:
    """Multiply val3 and val4."""
    state["result2"] = state["val3"] * state["val4"]
    state["message"] += (
        f"\nMultiplication performed on {state['val3']} and "
        f"{state['val4']}. Result: {state['result2']}"
    )  # noqa: E501
    return state


def divider1_node(state: AgentState) -> AgentState:
    """Divide val1 by val2."""
    if state["val2"] == 0:
        raise ValueError("Division by zero is not allowed.")
    state["result1"] = state["val1"] / state["val2"]
    state["message"] += (
        f"Division performed on {state['val1']} and "
        f"{state['val2']}. Result: {state['result1']}"
    )  # noqa: E501
    return state


def divider2_node(state: AgentState) -> AgentState:
    """Divide val3 by val4."""
    if state["val4"] == 0:
        raise ValueError("Division by zero is not allowed.")
    state["result2"] = state["val3"] / state["val4"]
    state["message"] += (
        f"\nDivision performed on {state['val3']} and "
        f"{state['val4']}. Result: {state['result2']}"
    )  # noqa: E501
    return state


def decide_next_node_router1(state: AgentState) -> str:
    """Route based on the first operation."""
    if state["operation1"] == "+":
        return "add_operation"
    elif state["operation1"] == "-":
        return "subtract_operation"
    elif state["operation1"] == "*":
        return "multiply_operation"
    elif state["operation1"] == "/":
        return "divide_operation"
    else:
        raise ValueError(f"Unknown operation: {state['operation1']}")


def decide_next_node_router2(state: AgentState) -> str:
    """Route based on the second operation."""
    if state["operation2"] == "+":
        return "add_operation"
    elif state["operation2"] == "-":
        return "subtract_operation"
    elif state["operation2"] == "*":
        return "multiply_operation"
    elif state["operation2"] == "/":
        return "divide_operation"
    else:
        raise ValueError(f"Unknown operation: {state['operation2']}")


def result_reporter_node(state: AgentState) -> AgentState:
    """Report the results."""
    print("-" * 40, "\n")
    print("\tRESULTS REPORT")
    print("-" * 40, "\n")
    print(f"Result 1: {state['result1']}")
    print(f"Result 2: {state['result2']}")
    print("-" * 40, "\n")
    print("Operations Steps:")
    print(state["message"])
    print("-" * 40, "\n")
    return state


###########
# Graph
###########

graph = StateGraph(state_schema=AgentState)

# Add nodes to the graph
graph.add_node("adder1", adder1_node)
graph.add_node("adder2", adder2_node)
graph.add_node("subtractor1", subtractor1_node)
graph.add_node("subtractor2", subtractor2_node)
graph.add_node("multiplier1", multiplier1_node)
graph.add_node("multiplier2", multiplier2_node)
graph.add_node("divider1", divider1_node)
graph.add_node("divider2", divider2_node)
graph.add_node("router1", lambda state: state)
graph.add_node("router2", lambda state: state)
graph.add_node("reporter", result_reporter_node)

# Add edges with conditional logic
graph.add_edge(START, "router1")
graph.add_conditional_edges(
    "router1",
    decide_next_node_router1,
    {
        "add_operation": "adder1",
        "subtract_operation": "subtractor1",
        "multiply_operation": "multiplier1",
        "divide_operation": "divider1",
    },
)

graph.add_edge("adder1", "router2")
graph.add_edge("subtractor1", "router2")
graph.add_edge("multiplier1", "router2")
graph.add_edge("divider1", "router2")

graph.add_conditional_edges(
    "router2",
    decide_next_node_router2,
    {
        "add_operation": "adder2",
        "subtract_operation": "subtractor2",
        "multiply_operation": "multiplier2",
        "divide_operation": "divider2",
    },
)

graph.add_edge("adder2", "reporter")
graph.add_edge("subtractor2", "reporter")
graph.add_edge("multiplier2", "reporter")
graph.add_edge("divider2", "reporter")
graph.add_edge("reporter", END)

app = graph.compile()


##############################################
# Displaying and Saving the graph structure
##############################################

"""
NOTE:
- Internet connection is required to display or save the graph.
- This will generate a PNG image of the graph structure and save it.
"""


# Save the graph structure
outpute_path = os.path.join("src", "ex4", "graph_structure_ex4.png")
app.get_graph().draw_mermaid_png(output_file_path=outpute_path)
print(f"Graph Structure saved as {outpute_path}")
print("\n---\n")


##################
# Execute Graph
##################
# This function will run the graph with an initial state and print the
# final state.
# You can modify the initial state as needed to test different scenarios.


def main(app: StateGraph):
    """Run the graph with an initial state."""
    # Define the initial state
    initial_state = AgentState(
        val1=10,
        val2=5,
        val3=7,
        val4=2,
        operation1="+",
        operation2="-",
        result1=0,
        result2=0,
        message="",
    )

    # Run the graph with the initial state
    final_state = app.invoke(initial_state)

    # Print the final state
    print("Final State:\n")
    printer.pprint(final_state)
    print("")


if __name__ == "__main__":
    # Run the main function with the compiled graph
    main(app)
