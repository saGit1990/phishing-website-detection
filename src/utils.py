import os

def print_header(message):
    print("\n" + "=" * len(message))
    print(message)
    print("=" * len(message) + "\n")


def ensure_artifacts_dir(path):
    """Ensure the parent directory of `path` exists.
    
    This extracts the directory portion of the path and creates it if needed.
    """
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        print(f"Created directory: {dirpath}")