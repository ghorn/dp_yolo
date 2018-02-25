all: state_transitions.pickle solution.pickle wat

state_transitions.pickle: make_state_transitions.py ocp_config.py
	./make_state_transitions.py

solution.pickle: state_transitions.pickle dp.py
	./dp.py

wat: plot_sol.py
	./plot_sol.py
