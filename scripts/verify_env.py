from pathlib import Path

from src.pipeline.bootstrap import run_verify_env
from src.pipeline.config import load_config


CONFIG_PATH = Path("configs/default.yaml")


if __name__ == "__main__":
    raise SystemExit(run_verify_env(config=load_config(CONFIG_PATH), config_path=CONFIG_PATH))
