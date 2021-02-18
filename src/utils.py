import os
import git
from git import Repo
from pathlib import Path

PROJ_ROOT = str(Path(__file__).parent.parent)

def savencommit(file):
    file = os.path.abspath(file)
    rep_dir = '.'
    if not PROJ_ROOT in file:
        print("File {} not in root directory {}".format(file, PROJ_ROOT))
        return
    file = file.split(PROJ_ROOT+'/')[1]
    repo = Repo(PROJ_ROOT)
    # repo.git.pull()
    
    return repo


def get_file(name, path='.'):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    if len(result)>1:
        print("Found multiple matches. Returning the first match")
    elif len(result)==0:
        print("No match")
        return None
    return result[0]

savencommit(__file__)
