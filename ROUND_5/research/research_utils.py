import json
import yaml

def load_queue(path="research/experiment_queue.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_state(state, path="research/experiment_state.json"):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def load_state(path="research/experiment_state.json"):
    with open(path, "r") as f:
        return json.load(f)