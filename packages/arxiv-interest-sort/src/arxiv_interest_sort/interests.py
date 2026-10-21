import dataclasses
import json
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class InterestTopic:
    name: str
    weight: int
    keywords: list[str]
    description: Optional[str] = None

    def asdict(self):
        return dataclasses.asdict(self)


def read_interests_from_file(file_path: str | Path) -> list[InterestTopic]:

    file_path = Path(file_path)
    if file_path.suffix in {".yml", ".yaml"}:
        with file_path.open("r") as f:
            data = yaml.safe_load(f)
    elif file_path.suffix == ".json":
        with file_path.open("r") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format. Please use .yml or .yaml")

    return [
        InterestTopic(
            name=topic["name"],
            weight=topic["weight"],
            keywords=topic["keywords"],
            description=topic.get("description"),
        )
        for topic in data
    ]


def write_interests_to_file(interests: list[InterestTopic], file_path: str | Path):
    file_path = Path(file_path)
    if file_path.suffix in {".yml", ".yaml"}:
        with file_path.open("w") as f:
            yaml.dump([interest.asdict() for interest in interests], f, sort_keys=False)
    elif file_path.suffix == ".json":
        with file_path.open("w") as f:
            json.dump(
                [interest.asdict() for interest in interests],
                f,
                sort_keys=False,
                indent=2,
            )
    else:
        raise ValueError("Unsupported file format. Please use .yml or .yaml")
