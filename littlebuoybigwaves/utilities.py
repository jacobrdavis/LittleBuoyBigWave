import toml


def get_config():
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    return config