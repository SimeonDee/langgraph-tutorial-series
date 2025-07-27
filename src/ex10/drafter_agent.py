"""
Implementing Simple Human-in-the-Loop (HITL) ReAct Agents
(Agent with Tools) using LangGraph
=========================================================

Goal:
To learn how to build a Simple Collaborative (HITL) ReAct agent
using LangGraph.

Objectives:
- How to create a "ReAct Graph" with integrated HITL.
- Work with different types of "Messages" such as "ToolMessage"
- Integrate tool-use into Agents with LangGraph.


## Scenerio:

# Problem:

CEO:
Our company is not working efficiently! We spend way too much time
drafting  documents and this needs to be fixed!

# Your task:

For the company, you need to create an AI Agentic System that can
"speed up drafting documents , emails, etc". The AI Agentic System
should have "Human-AI Collaboration" meaning the Human should be able to
provide "continuous feedback" and the AI System should stop when the Human
is happy with the draft. The System should be fast and be able to save
the drafts.
"""

import os
from pprint import PrettyPrinter
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


from dotenv import load_dotenv

load_dotenv()
# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

printer = PrettyPrinter(indent=3, sort_dicts=False)
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_retries=3,
    max_tokens=1000,
    request_timeout=60,
    verbose=True,
)
DOCUMENT_CONTENT = ""  # global variable to hold draft contents
DRAFT_FOLDER = os.path.join("src", "ex10")


###############
# State
###############


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


###############
# Tools
###############


@tool
def update(content: str) -> str:
    """Updates document with the provided content

    Args:
        content (str): the content to use.
    """
    global DOCUMENT_CONTENT
    DOCUMENT_CONTENT = content
    return (
        "Document has been updated successfuly. \n"
        f"The current content is: \n\n---\n\n{DOCUMENT_CONTENT}\n\n---\n\n"
    )


@tool
def save(filename: str) -> str:
    """Saves the current document to a text file and finish the process.

    Args:
        filename (str): the name of the file to save contents to.
    """
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    filename = os.path.join(DRAFT_FOLDER, filename)

    if not DOCUMENT_CONTENT:
        return "There are no content to save yet."

    try:
        with open(filename, "w") as file:
            file.write(DOCUMENT_CONTENT)
        print(f"Document saved into {filename}")
        return f"Document has been saved successfully to '{filename}'"
    except Exception as e:
        return f"Error Saving document: {e}"


tools = [update, save]
model = model.bind_tools(tools)


#############
# Nodes
#############


def drafter_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
        You are Drafter. A helpful writing assistant. You are going to
        user update and modify documents.

        - If the user wants to update or modify content, use the 'update'
          tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save'
          tool.
        - Make sure to always show the current document state after
          modification.

        The current document content is: {DOCUMENT_CONTENT}
    """
    )

    if not state["messages"]:
        user_input = (
            "I'm ready to help you update a document. "
            "What would you like to create?"  # no-qa
        )
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")

    user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"... USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> AgentState:
    """Determins if we should continue or end the conversation."""
    messages = state["messages"]

    if not messages:
        return "continue"

    # Look for the most recent tool message...
    for message in reversed(messages):
        # check if it's a ToolMessage resulting from save operation
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"

    return "continue"


# Utility Function
def print_messages(messages: list):
    """Custom print func."""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n...TOOL RESULT: {message.content}")


###############
# Graph
###############

graph = StateGraph(state_schema=AgentState)

# Add nodes
graph.add_node("agent", drafter_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


###########################
# Save the graph structure
###########################


output_path = os.path.join("src", "ex10", "graph_structure_ex10.png")
app.get_graph().draw_mermaid_png(output_file_path=output_path)
print(f"Graph Structure saved as {output_path}")
print("\n---\n")


################
# Execute Graph
################


def run_drafter_agent():
    """Runs the agent graph."""
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_drafter_agent()
