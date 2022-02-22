from pathlib import Path
import os


def get_project_root():
    '''
    for setting the root of the project independent of OS.
    :return: root of the project
    '''

    return os.path.abspath(Path(__file__).parent.parent)
