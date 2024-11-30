# Define the Python command
PYTHON = python

# Define the main script
RUN_SCRIPT = src/main.py

# Define the evaluation script.
EVAL_SCRIPT = evaluation/UTILITY.py

# Define the Metrics script.
METRIC_SCRIPT = results/getMetrics.py

# Define a phony target for running the script
.PHONY: run eval result

# Default target to run the script with specified arguments
run:
	$(PYTHON) $(RUN_SCRIPT) $(args)

# Target for pooling methods
eval:
	$(PYTHON) $(EVAL_SCRIPT) "./data/SBMnet_dataset/" $(args)

# Results
result:
	$(PYTHON) $(METRIC_SCRIPT) $(args)
