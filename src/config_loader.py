import yaml

def load_config(path='config.yaml'):
    """Load configurations from a YAML file.
    
    Args:
        path (str): path to the YAML configuration files
    """

    with open(path,'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__== "__main__":
    config = load_config('config.yaml')
    print(config)