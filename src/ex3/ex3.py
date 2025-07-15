import os
from typing import TypedDict, List
from langgraph.graph import StateGraph
from pprint import PrettyPrinter

"""
Demonstrate How to Implement simple Sequential Graph in LangGraph

Main Goal:
- Learn how to create and handle multiple Nodes in LangGraph.

Objectives:
- Create multiple Nodes that sequentially process and update \
    different parts of the state.
- Connect Nodes together to form a sequential flow.
- Invoke the Graph and see how the state is transformed step-by-step.
"""

"""
Problem Description:
You are tasked with implementing a multi-node graph.

Requirements:
- Accept user's 'name', 'age' and list of 'skills'
- Pass the state through three nodes that will:
  1. First node: Personalize the name field with a greeting.
  2. Second node: Describe the user's age.
  3. Third node: List the user's skills in a formatted string.
- The final output in the 'message' field should be a \
    'combined message' in the format below:

    Output: "Ade, welcome to the system! You are 30 years old \
        and your skills are: Python, Java, and C++."
"""

printer = PrettyPrinter(indent=3, sort_dicts=False)


###########
# State
###########


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    name: str  # User's name
    age: int  # User's age
    skills: List[str]  # List of user's skills
    message: str  # Message to display to the user


###########
# Nodes
###########


def greet_user_node(state: AgentState) -> AgentState:
    """
    A node that personalizes the name field with a greeting.
    """
    state["message"] = f"{state['name']}, welcome to the system!"
    return state


def describe_age_node(state: AgentState) -> AgentState:
    """
    A node that describes the user's age.
    """
    state["message"] += f" You are {state['age']} years old."
    return state


def list_skills_node(state: AgentState) -> AgentState:
    """
    A node that lists the user's skills in a formatted string.
    """
    skills_str = (
        ", ".join(state["skills"][:-1]) + " and " + state["skills"][-1]
        if len(state["skills"]) > 1
        else (state["skills"][0] if state["skills"] else "no skills")
    )  # noqa: E501
    state["message"] += f" and your skills are: {skills_str}."
    return state


###########
# Graph
###########


graph = StateGraph(state_schema=AgentState)

# Add nodes
graph.add_node("greeter", greet_user_node)
graph.add_node("age_describer", describe_age_node)
graph.add_node("list_skills", list_skills_node)

# Add edges
graph.add_edge("greeter", "age_describer")
graph.add_edge("age_describer", "list_skills")

graph.set_entry_point("greeter")
graph.set_finish_point("list_skills")

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


# Displaying the graph as an image (will only disolay in Jupyter Notebook)
# graph_image = app.get_graph().draw_mermaid_png()
# display(Image(graph_image))

# Save the graph structure
outpute_path = os.path.join("src", "ex3", "graph_structure_ex3.png")
app.get_graph().draw_mermaid_png(output_file_path=outpute_path)
print(f"Graph Structure saved as {outpute_path}")
print("\n---\n")


##################
# Execute Graph
##################


def main(app: StateGraph):
    # setting up initial state
    skills = ["Java", "Go", "Docker", "CI/CD", "Python"]
    initial_state = AgentState(name="Debo", age=25, skills=skills)

    result = app.invoke(initial_state)

    print("\n---\n")
    print(result["message"])
    print("\n---\n")
    print("Current Agent State:\n")
    printer.pprint(result)
    print("\n---\n")


if __name__ == "__main__":
    main(app)
