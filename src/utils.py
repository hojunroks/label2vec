import os

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
