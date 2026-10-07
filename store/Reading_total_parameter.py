"""Print and save the total parameter count of the configured DOAMambaNet."""

from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.mamba_cnn import DOAMambaNet


def main():
    with (PROJECT_ROOT / "configs" / "modanet.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    model = DOAMambaNet(
        dataset_config=cfg["dataset"],
        model_config=cfg["model"],
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    result = f"Total parameters: {total_parameters}"
    print(result)

    output_path = PROJECT_ROOT / "results" / "total_parameters.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(result + "\n")


if __name__ == "__main__":
    main()
