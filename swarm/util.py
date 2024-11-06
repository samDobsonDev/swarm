import inspect
from datetime import datetime

def debug_print(debug: bool, *args: str) -> None:
    if not debug: return  # If debugging is not enabled, exit the function
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time formatted as a timestamp
    message = " ".join(map(str, args))  # Convert all arguments to strings and join them into a single message
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")  # Print the timestamp and message with color formatting

def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {  # Map Python types to JSON schema types
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    try:
        signature = inspect.signature(func)  # Get the function's signature
    except ValueError as e:
        raise ValueError(  # Raise an error if the signature cannot be obtained
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    parameters = {}  # Initialize a dictionary to hold parameter details
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")  # Get the parameter type
        except KeyError as e:
            raise KeyError(  # Raise an error if the type annotation is unknown
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}  # Add parameter type to the dictionary
    required = [
        param.name  # Extract the name of the parameter
        for param in signature.parameters.values()  # Iterate over each parameter in the function's signature
        if param.default is param.empty  # Include only parameters without default values
    ]
    return {  # Return the function signature as a JSON-serializable dictionary
        "type": "function",
        "function": {
            "name": func.__name__,  # Function name
            "description": func.__doc__ or "",  # Function description from docstring
            "parameters": {
                "type": "object",
                "properties": parameters,  # Parameter details
                "required": required,  # Required parameters
            },
        },
    }
