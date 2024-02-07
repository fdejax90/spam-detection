from pathlib import Path

from preprocessing import preprocess_sms
from training import train_baselines

if __name__ == "__main__":
    file_path = Path("/opt/static/processed/dataset.csv")
    if not file_path.exists():
        preprocess_sms("/opt/static/raw/dataset.csv")

    Path("'/opt/static/outputs/csv").mkdir(parents=True, exist_ok=True)

    # Train baseline models
    train_baselines(
        file_path,
        [0.8],
        0,
        "test",
    )
