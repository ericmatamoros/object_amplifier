"""Define settings"""
import yaml

from object_amplifier import CONFIG_PATH


def U2NetSettings():
    with open(CONFIG_PATH / "config.yaml", "r") as yaml_file:
        return yaml.safe_load(yaml_file)
    
