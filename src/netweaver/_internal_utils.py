import csv
import datetime
import os


def get_datetime() -> datetime.datetime:
    """Returns the current local date and time as a datetime object.

    This function provides a convenient way to access the current timestamp.

    Returns:
        datetime.datetime: The current local date and time.
    """
    return datetime.datetime.now()

def get_dirname(path_file: str):
    """Returns the directory name of the specified file path.

    This function extracts and returns the directory component from a given file path.

    Args:
        path_file (str): The file path from which to extract the directory name.

    Returns:
        str: The directory name of the provided file path.
    """
    return os.path.dirname(path_file)


def get_cwd() -> str:
    """Returns the current working directory as a string.

    This function retrieves the absolute path of the current working directory.

    Returns:
        str: The current working directory path.
    """
    return os.getcwd()


def join_path(*args):
    """Joins one or more path components intelligently.

    This function combines multiple path components into a single path string.

    Args:
        *args: Components to be joined into a path.

    Returns:
        str: The joined path string.
    """
    return os.path.join(*args)


def _create_directory(path: str) -> bool:
    """Creates a directory at the specified path if it does not already exist.

    This function attempts to create the directory and returns True if successful, or False if an error occurs.

    Args:
        path (str): The path where the directory should be created.

    Returns:
        bool: True if the directory was created or already exists, False otherwise.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating directory: {e}")
        return False


def create_log_dir(log_path: str = None, now: datetime.datetime = None) -> str | None:
    """Creates a timestamped model training log directory and returns its path.

    This function constructs a directory path using the provided log path and current timestamp, creates the directory, and returns the path if successful.

    Args:
        log_path (str, optional): The base path where the log directory will be created. Defaults to the current working directory.
        now (datetime.datetime, optional): The datetime object used for timestamping the directory name.

    Returns:
        str or None: The path to the created log directory, or None if creation failed.
    """
    path_model_log = join_path(log_path, "logs", "model_training", f"model-{now:%Y%m%d-%H%M%S}")
    return path_model_log if _create_directory(path_model_log) else None


def create_log_file(path_model_log: str = None, type: str = None, field_names: list = None, now: datetime.datetime = None) -> str:
    """Creates a CSV log file with a timestamped name and specified field names.

    This function generates a new CSV file in the given log directory, writes the header row, and returns the file path.

    Args:
        path_model_log (str, None): The directory where the log file will be created.
        type (str, optional): The type of log, used in the file name.
        field_names (list, optional): The list of field names for the CSV header.
        now (datetime.datetime, optional): The datetime object used for timestamping the file name.

    Returns:
        str: The path to the created log file.
    """
    if field_names is None:
        field_names = []
    file_name = f"Log-{type}-{now:%Y%m%d-%H%M%S}.csv"
    path_file = join_path(path_model_log, file_name)

    try:
        with open(path_file, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
            csv_writer.writeheader()
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Error writing log file '{path_file}': {e}")
    return path_file


def append_log_file(path_file: str = None, field_names: list = None, field_values: dict = None) -> None:
    """Appends a row of data to an existing CSV log file.

    This function writes the provided field values as a new row in the specified CSV file using the given field names.

    Args:
        path_file (str): The path to the CSV log file.
        field_names (list): The list of field names for the CSV header.
        field_values (dict): The dictionary of values to append as a row.
    """
    try:
        with open(path_file, "a", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
            csv_writer.writerow(field_values)
            csv_file.flush()
    except OSError as e:
        print(f"Error writing log file '{path_file}': {e}")
