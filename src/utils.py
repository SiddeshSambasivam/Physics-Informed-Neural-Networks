import json


def read_config(filepath: str) -> dict:

    with open(filepath) as file_:
        values = json.load(file_)

    return values
