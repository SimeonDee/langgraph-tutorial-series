VENV_NAME = .venv
MODULE_NAME = src/ex10/drafter_agent.py

# create venv
venv:
	pip install uv
	uv venv $(VENV_NAME)

# install dependencies
install:
	uv add -r requirements.txt

# run agent
run_ex10:
	uv run src/ex10/drafter_agent.py
