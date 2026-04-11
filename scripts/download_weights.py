from pathlib import Path

from src.pipeline.config import load_config
from src.pipeline.weights import run_download_weights


CONFIG_PATH = Path("configs/default.yaml")


if __name__ == "__main__":
    raise SystemExit(
        run_download_weights(
            config=load_config(CONFIG_PATH),
            config_path=CONFIG_PATH,
            url_override=None,
            force=False,
        )
    )
