import os
import errno


def get_files(folder, name_filter=None):
    """
    Helper function that returns the list of files in a specified folder.
    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    filtered_files = []

    # Explore the directory tree to get files.
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_filter is not None and not name_filter(file):
                continue
            else:
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def get_dirs(folder, name_filter=None):
    """
    Helper function that returns the list of directories in a specified folder.
    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    filtered_dirs = []

    # Explore the directory tree to get files.
    for path, dirs, _ in os.walk(folder):
        dirs.sort()
        for dir in dirs:
            if name_filter is not None and not name_filter(dir):
                continue
            else:
                full_path = os.path.join(path, dir)
                filtered_dirs.append(full_path)

    return filtered_dirs


def ensure_dir(directory):
    """
    Ensure the given directory exists. If not, create it.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise exc
