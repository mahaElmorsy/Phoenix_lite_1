"""phoenixlit.cli

Command line for Phoenix Lite version 0.1.0

Commands:
- run
- collect-gesture --label YES ====> label can be changed to the sign language I want to be detected
- train-gesture --dataset-dir data/gesture --out-model models/gesture.joblib
"""

from pathlib import Path
import typer

from .auth import PinConfig, verify_pin
from .config import load_config
from .runtime import run_app
from .collectors import run_collect_gesture
from .trainers import train_gesture

app = typer.Typer(add_completion=False, help="Phoenix Lit 0.1 CLI")


@app.command()
def run(
    config: Path = typer.Option("config.yaml", "-c", "--config", exists=True, dir_okay=False),
):
    """Run the main camera UI."""
    cfg = load_config(config)

    pin_cfg = PinConfig(
        pin_file=Path(str(cfg.get("auth.pin_file", "data/auth/pin.json"))),
        failed_file=Path(str(cfg.get("auth.failed_file", "data/auth/failed.txt"))),
        max_attempts=int(cfg.get("auth.max_attempts", 3)),
    )

    warning = verify_pin(pin_cfg)
    if warning:
        print(f"[Phoenix] {warning}")

    run_app(cfg)


@app.command("collect-gesture")
def collect_gesture(
    label: str = typer.Option(..., "--label", "-l", help="Gesture label to collect (e.g., YES/NO/STOP/HELP/PAIN/NONE)"),
    config: Path = typer.Option("config.yaml", "-c", "--config", exists=True, dir_okay=False),
):
    """Collect training samples for gestures."""
    cfg = load_config(config)
    run_collect_gesture(cfg, label=label)


@app.command("train-gesture")
def train_gesture_cmd(
    dataset_dir: Path = typer.Option("data/gesture", "--dataset-dir"),
    out_model: Path = typer.Option("models/gesture.joblib", "--out-model"),
):
    """Train a gesture model from the collected samples in data/gesture"""
    out = train_gesture(dataset_dir=dataset_dir, out_model=out_model)
    print(f"[Phoenix] Gesture model saved to: {out}")


if __name__ == "__main__":
    app()
