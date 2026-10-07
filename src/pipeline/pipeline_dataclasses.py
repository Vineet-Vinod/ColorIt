from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightDownload:
    name: str
    repo_id: str
    url: str
    destination: Path
