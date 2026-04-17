import logging
import logging.config
import yaml

def setup_logging():
    config_path = "app/config/logging_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)