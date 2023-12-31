import os
from typing import List

def get_list_of_files(dirName: str) -> List[str]:
    """test

    Args:
        dirName (str): _description_

    Returns:
        List[str]: _description_
    """
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles: List[str] = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            if not (("metadata" in entry) or ("DS_Store" in entry)):
                allFiles.append(fullPath)

    return sorted(allFiles)