F1 Qualifying & Race Prediction Tool

This repository contains a Python script that generates Formula 1 qualifying and race predictions using historical session data. The goal is to have a script that can be run before any race weekend and produce updated predictions without manually changing seasons, drivers, teams, or race names.

The script relies on the FastF1 library to pull official timing and results data and builds a lightweight statistical model on top of it.

Features

Automatically detects the current F1 season

Pulls data from the most recent completed seasons

Weights recent races more heavily than older ones

Automatically identifies the next upcoming Grand Prix

No hard-coded drivers, teams, or calendar dates

Predicts:

Qualifying order (Q3-style pace ranking)

Race finishing order (pace-based proxy)

Once set up, the script can be run without modification year to year.

How It Works

At a high level, the script does the following:

Determines the current year and loads the last N completed seasons of qualifying data

Pulls Q1, Q2, and Q3 times from completed races using FastF1

Trains a regression model to learn how Q1/Q2 performance translates to Q3 pace

Applies sample weighting so recent races influence predictions more

Detects the next race on the official F1 calendar

Builds the current driver/team grid dynamically from recent data

Predicts qualifying pace using recent driver and team performance

Uses the predicted qualifying order as a simple race finishing prediction

This is intentionally a data-driven heuristic, not a full race simulator.

Requirements

Python 3.9+

fastf1

pandas

numpy

scikit-learn

Install dependencies with:

pip install fastf1 pandas numpy scikit-learn

Usage

Run the script directly:

python f1_predictions.py


On first run, FastF1 will download and cache data locally. Subsequent runs will be faster.

The script outputs:

Model performance metrics

The next upcoming Grand Prix

Predicted qualifying order (top 10)

Predicted race finishing order

No command-line arguments are required.

Design Decisions

No static configuration: Seasons, drivers, teams, and race rounds are discovered dynamically.

Recency bias: Recent races matter more than older ones.

Simplicity over complexity: The model favors transparency and robustness over overfitting.

Extensibility: The structure allows future additions like track-specific models, weather effects, or race simulations.

Limitations

Does not simulate race strategy, tire degradation, safety cars, or DNFs

Weather is not explicitly modeled

Race predictions are pace-based, not strategy-based

Accuracy depends on available historical data

This is meant to provide reasonable, current-form predictions, not guaranteed results.

Data Source

All timing and results data is provided by the FastF1 library, which sources official Formula 1 timing data.

Formula 1 data remains the property of Formula One Management.

Future Improvements

Possible extensions include:

Track-specific pace normalization

Weather and temperature effects

Monte Carlo race simulations

Strategy and pit stop modeling
