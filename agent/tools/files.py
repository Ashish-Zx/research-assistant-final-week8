# agent/tools/files.py
import os

WORKSPACE = os.path.join(os.getcwd(), "workspace")
os.makedirs(WORKSPACE, exist_ok=True)


def _safe_path(filename: str) -> str:
    """Prevent directory traversal attacks."""
    base = os.path.basename(filename)
    return os.path.join(WORKSPACE, base)


def read_file(filename: str) -> str:
    """Read the contents of a file in the workspace."""
    path = _safe_path(filename)
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File '{filename}' not found."
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filename: str, content: str) -> str:
    """Write content to a file in the workspace. Overwrites if exists."""
    path = _safe_path(filename)
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File '{filename}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"


def list_files() -> str:
    """List all files in the workspace."""
    try:
        files = os.listdir(WORKSPACE)
        if not files:
            return "No files in workspace."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"
