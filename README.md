# Credit Scorecard

A credit scorecard (WoE/IV + logistic regression) on the Give Me Some Credit dataset, served through a FastAPI scoring service. This is a method demonstration on US data, not a model for Indian lending.

## Setup

Dependencies are pinned in `pyproject.toml`. Newer numpy and pandas builds are blocked by Application Control on the development laptop, so this project uses older, widely distributed releases.

    python -m venv C:\venvs\sc2
    C:\venvs\sc2\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"

Run tools as `python -m pytest` and `python -m ruff`. On machines with Application Control, create the venv on C: and avoid bare `.exe` launchers.

## Data

Download `cs-training.csv` from the Give Me Some Credit competition on Kaggle and place it in `data/raw/`. The data folder is gitignored.
