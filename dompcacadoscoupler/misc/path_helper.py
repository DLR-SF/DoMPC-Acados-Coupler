from pathlib import Path


def get_project_root() -> Path:
    """Get path of the project root folder.

    Returns:
        Path: Folder path to root.
    """
    # From: https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure
    # absolute is used because on some OSs the Path(__file__) may return the relative path.
    return Path(__file__).absolute().parent.parent
