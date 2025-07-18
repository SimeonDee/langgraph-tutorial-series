import os
import random
from langgraph.graph import StateGraph, START, END
from pprint import PrettyPrinter
from typing import List, TypedDict

"""
Demonstrating Looping Graph Implementation - Part2

Main Goal:
- Learn how to code 'Looping Logic'.

Objectives:
- Implement 'looping logic' to route the flow of data back to the node.
- Create a 'conditional edge' to handle decision making and
    control graph flow.
"""

"""
Problem Description:
    You are tasked with implementing "Guess A Number Game"
    with integrated hinter (Automatic 'higher' or 'lower' hinter).

Requirements:
- Set the bound to be between 1 and 20.
- The graph has to keep guessing (max number of guesses is 7), \
    where if the guess is correct, then it stops, if not, we keep looping
    until we hit the max limit of 7.
- Each time the user guesses a number, the 'hint node should say \
    higher or lower' and the graph should account for this information and \
        guess the next guess accordingly.

- input:
{"playername":"Student", "guesses":[], attempts:0, \
    "lower_bound"=1, "upper_bound": 20, "message":""}
- output:
{"playername":"Student", "guesses":[5, 10, 15], attempts:3, \
    "lower_bound":1, "upper_bound":20, "message":"
    "Hello Student, your guesses were: 5, 10, 15. \
    You made 3 attempts. The number was 15."}
"""


printer = PrettyPrinter(indent=3, sort_dicts=False)


###########
# State
###########


class AgentState(TypedDict):
    """
    Represents the state of the graph.
    """

    playername: str  # Player's name
    target_number: int  # The number to guess
    guesses: List[int]  # List of player's guesses
    attempts: int  # Number of attempts made
    max_attempts: int  # Maximum number of attempts allowed
    lower_bound: int  # Lower bound for guessing
    upper_bound: int  # Upper bound for guessing
    message: str  # Message to display to the user


############
# Nodes
############


def greet_user_node(state: AgentState) -> AgentState:
    """
    Greet the user.
    """
    state["message"] += (
        f"Hello {state['playername']}, " "let's play a guess game!\n"
    )  # noqa: E501
    random_number = random.randint(state["lower_bound"], state["upper_bound"])
    state["target_number"] = random_number
    state["message"] += (
        f"The number to guess is between {state['lower_bound']} and "
        f"{state['upper_bound']}.\n"
    )  # noqa: E501
    return state


def guess_number_node(state: AgentState) -> AgentState:
    """
    Guess a number between the lower and upper bounds.
    """
    if state["attempts"] >= state["max_attempts"]:
        state["message"] += "Maximum attempts reached. Game over!\n"
        print("Maximum attempts reached. Game over!\n")
        return state
    guess = int(
        input(
            f"Guess a number between {state['lower_bound']} and "
            f"{state['upper_bound']}: "
        )
    )  # noqa: E501
    state["message"] += f"Your guess: {guess}\n"
    # Ensure the guess is within bounds
    if guess < state["lower_bound"] or guess > state["upper_bound"]:
        state["message"] += (
            f"Invalid guess! Please guess a number between "
            f"{state['lower_bound']} and {state['upper_bound']}.\n"
        )
        print(
            f"Invalid guess! Please guess a number between "
            f"{state['lower_bound']} and {state['upper_bound']}.\n"
        )
        return state
    state["guesses"].append(guess)
    state["attempts"] += 1
    state["message"] += (
        f"Attempt {state['attempts']}: "
        f"{state['max_attempts'] - state['attempts']} remaining\n"
    )  # noqa: E501

    return state


def hint_decider(state: AgentState) -> str:
    """
    Provide a hint based on the last guess.
    """
    if state["attempts"] == 0:
        return "No guesses made yet"

    if state["attempts"] >= state["max_attempts"]:
        state["message"] += (
            "Game over! No more hints available.\n "
            f"Maximum attempts reached {state['attempts']} attempts.\n"  # noqa: E501
        )  # noqa: E501
        print("Game over! No more hints available.")
        return "exit"  # Game over, no more hints available

    last_guess = state["guesses"][-1]

    if last_guess < state["target_number"]:
        state["message"] += "Hint: Lower!\n"
        state[
            "message"
        ] += f"{state['max_attempts'] - state['attempts']} attempts left.\n"
        print("Hint: Lower!")
        print(f"{state['max_attempts'] - state['attempts']} attempts left.\n")
        return "lower"

    elif last_guess > state["target_number"]:
        state["message"] += "Hint: Higher!\n"
        print("Hint: Higher!")
        return "higher"

    else:
        state["message"] += "Congratulations! You've guessed the number!\n"
        print("Congratulations! You've guessed the number!")
        state["message"] += (
            f"Your guesses were: {', '.join(map(str, state['guesses']))}. "
            f"You made {state['attempts']} attempts. The number was "
            f"{state['target_number']}.\n"
        )
        return "correct"


def farewell_user_node(state: AgentState) -> AgentState:
    """
    Farewell the user and display the guesses.
    """
    if state["guesses"][-1] != state["target_number"]:
        state[
            "message"
        ] += "You didn't guess the number. Better luck next time!\n"  # noqa: E501
        print("You didn't guess the number. Better luck next time!\n")
    else:
        state["message"] += "Congratulations! You guessed right\n"
        print("Congratulations! You guessed right\n")

    state["message"] += f"\nGoodbye, {state['playername']}!\n\n"
    state["message"] += (
        f"Your guesses were: {', '.join(map(str, state['guesses']))}. "
        f"You made {state['attempts']} attempts. The number was "
        f"{state['target_number']}.\n"
    )
    return state


###########
# Graph
###########

graph = StateGraph(state_schema=AgentState)

# Add nodes to the graph
graph.add_node("greeter", greet_user_node)
graph.add_node("number_guesser", guess_number_node)
graph.add_node("farewell", farewell_user_node)

# Add edges with conditional logic
graph.add_edge(START, "greeter")
graph.add_edge("greeter", "number_guesser")
graph.add_conditional_edges(
    "number_guesser",
    hint_decider,  # The hint decider node
    {
        "lower": "number_guesser",  # Loop back for lower hint
        "higher": "number_guesser",  # Loop back for higher hint
        "correct": "farewell",  # End the game if correct guess
        "No guesses made yet": "number_guesser",  # No guesses made yet
        "exit": "farewell",  # Exit the game if max attempts reached
    },
)
# Add the farewell node to the end of the graph
graph.add_edge("farewell", END)

app = graph.compile()


################
# Main function to run the graph
################


def main(app: AgentState, initial_state: AgentState) -> None:
    """
    Run the graph with the initial state.
    """
    # Print the initial state
    print("\nInitial State:")
    printer.pprint(initial_state)
    print("\n---\n")

    # Run the graph
    final_state = app.invoke(initial_state)

    # Print the final state
    print("\nFinal State:")
    printer.pprint(final_state)
    print("\n---\n")

    # Print the message
    print("Message:\n")
    print(final_state["message"])
    print("\n---\n")


if __name__ == "__main__":
    # Setting up initial state
    initial_state = AgentState(
        playername="Student",
        target_number=0,
        guesses=[],
        attempts=0,
        max_attempts=7,
        lower_bound=1,
        upper_bound=20,
        message="",
    )

    main(app, initial_state)

    # Save the graph structure
    output_path = os.path.join("src", "ex6", "graph_structure_ex6.png")
    app.get_graph().draw_mermaid_png(output_file_path=output_path)
    print(f"Graph Structure saved as {output_path}")
    print("\n---\n")
