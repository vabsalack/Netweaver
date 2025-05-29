import csv
import datetime
import os


def get_datetime() -> datetime.datetime:
    """
    Returns the current local date and time as a datetime object.

    Parameters
    ----------
    None

    Returns
    -------
    datetime.datetime
        The current local date and time.
    """
    return datetime.datetime.now()


def get_dirname(path_file: str) -> str:
    """
    Returns the directory name of the specified file path.

    Parameters
    ----------
    path_file : str
        The file path from which to extract the directory name.

    Returns
    -------
    str
        The directory name of the provided file path.
    """
    return os.path.dirname(path_file)


def get_cwd() -> str:
    """
    Returns the current working directory as a string.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The current working directory path.
    """
    return os.getcwd()


def join_path(*args) -> str:
    """
    Joins one or more path components intelligently.

    Parameters
    ----------
    *args
        Components to be joined into a path.

    Returns
    -------
    str
        The joined path string.
    """
    return os.path.join(*args)


def _create_directory(path: str) -> bool:
    """
    Creates a directory at the specified path if it does not already exist.

    Parameters
    ----------
    path : str
        The path where the directory should be created.

    Returns
    -------
    bool
        True if the directory was created or already exists, False otherwise.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating directory: {e}")
        return False


def create_log_dir(log_path: str = None, now: datetime.datetime = None) -> str | None:
    """
    Creates a timestamped model training log directory and returns its path.

    Parameters
    ----------
    log_path : str, optional
        The base path where the log directory will be created. Defaults to the current working directory.
    now : datetime.datetime, optional
        The datetime object used for timestamping the directory name.

    Returns
    -------
    str or None
        The path to the created log directory, or None if creation failed.
    """
    path_model_log = join_path(log_path, "logs", "model_training", f"model-{now:%Y%m%d-%H%M%S}")
    return path_model_log if _create_directory(path_model_log) else None


def create_log_file(
    path_model_log: str = None,
    type: str = None,
    field_names: list = None,
    now: datetime.datetime = None,
) -> str:
    """
    Creates a CSV log file with a timestamped name and specified field names.

    Parameters
    ----------
    path_model_log : str, optional
        The directory where the log file will be created.
    type : str, optional
        The type of log, used in the file name.
    field_names : list, optional
        The list of field names for the CSV header.
    now : datetime.datetime, optional
        The datetime object used for timestamping the file name.

    Returns
    -------
    str
        The path to the created log file.
    """
    if field_names is None:
        field_names = []
    file_name = f"Log-{type}-{now:%Y%m%d-%H%M%S}.csv"
    path_file = join_path(path_model_log, file_name)

    try:
        with open(path_file, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
            csv_writer.writeheader()
    except OSError as e:
        print(f"Error writing log file '{path_file}': {e}")
    return path_file


def append_log_file(
    path_file: str = None,
    field_names: list = None,
    field_values: dict = None,
) -> None:
    """
    Appends a row of data to an existing CSV log file.

    Parameters
    ----------
    path_file : str, optional
        The path to the CSV log file.
    field_names : list, optional
        The list of field names for the CSV header.
    field_values : dict, optional
        The dictionary of values to append as a row.

    Returns
    -------
    None
    """
    try:
        with open(path_file, "a", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
            csv_writer.writerow(field_values)
            csv_file.flush()
    except OSError as e:
        print(f"Error writing log file '{path_file}': {e}")
