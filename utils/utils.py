
def get_name(path:str):
    filename = path.split("/")[-1]
    return ".".join(filename.split('.')[:-1])


def get_path(path:str):
    return "/".join(path.split("/")[:-1])

