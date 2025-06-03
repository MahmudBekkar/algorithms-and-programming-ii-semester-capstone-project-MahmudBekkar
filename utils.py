def validate_input(data):
    """
    Validate the input data for the algorithm.
    Return True if valid, else False.
    """
    if data is None:
        return False
    if not isinstance(data, list):
        return False
    return True

def print_debug(message):
    """
    Utility function to print debug messages.
    """
    print(f"[DEBUG]: {message}")

def load_json(filename):
    """
    Load JSON data from a file.
    """
    import json
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    """
    Save data to a JSON file.
    """
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

