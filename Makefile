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
run_ex1:
	uv run src/ex1/ex1.py

run_ex2:
	uv run src/ex2/ex2.py

run_ex3:
	uv run src/ex3/ex3.py

run_ex4:
	uv run src/ex4/ex4.py

run_ex5:
	uv run src/ex5/ex5.py

run_ex6:
	uv run src/ex6/ex6.py

run_ex7:
	uv run src/ex7/ex7.py

run_ex8:
	uv run src/ex8/ex8.py

run_ex9:
	uv run src/ex9/ex9.py

run_ex10:
	uv run src/ex10/drafter_agent.py
