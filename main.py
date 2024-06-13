import argparse
import os
import torch
import yaml
from datetime import datetime
from src.inference import inference

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config.yaml", type=str, help="Path to the YAML configuration file")
    return parser

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def test(config):
    # Inference
    print("Inference start...")
    torch.cuda.empty_cache()
    inference(config)

    print("[!!] Inference successfully complete.")

if __name__ == "__main__":
    arg_parser = config()
    args = arg_parser.parse_args()

    # Load config from YAML file
    config_dict = load_config_from_yaml(args.config_file)

    # Create Namespace object from the config dict
    config = argparse.Namespace(**config_dict)

    # Set the device attribute
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config.device = torch.device('cpu')
    config.timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    print("Number of CUDA devices", torch.cuda.device_count())
    test(config)
