F1 Race & Qualifying Prediction Script


This project is a Python script I put together to generate Formula 1 qualifying and race predictions using historical data. The idea is that I can run this before a race weekend and get a reasonable prediction without manually updating drivers, teams, or seasons.

The script uses the FastF1 library to pull official session data and builds a lightweight model on top of it.

This is not meant to be a perfect simulator — it’s more of a data-driven prediction tool that updates itself as the season progresses.


What This Script Does

Automatically detects the current F1 season

Pulls data from the last couple of completed seasons

Weights recent races more heavily

Automatically detects the next upcoming Grand Prix

Uses recent qualifying performance to predict:

Qualifying order (Q3-style ranking)

Race finishing order (simple proxy based on pace)

No hard-coded drivers, teams, or years

Once it’s set up, I just run it and it figures everything out on its own.


What It Uses

Python 3

fastf1

pandas

numpy

scikit-learn

FastF1 handles all the official F1 timing and results data, which makes life way easier.


How It Works (High Level)

Figures out what year it is and what races have already happened

Pulls qualifying data (Q1, Q2, Q3) from completed races

Trains a simple regression model to understand pace trends

Applies more weight to newer races

Detects the next race on the calendar

Predicts qualifying pace and sorts drivers accordingly

Uses that order as a rough race finishing prediction

There’s some randomness added so the output doesn’t look unrealistically exact.


How to Run It

Install dependencies first:

pip install fastf1 pandas numpy scikit-learn


Then just run:

python f1_predictions.py



The script will:

Cache data locally (first run is slower)

Print model stats

Print qualifying predictions

Print race finishing predictions

No arguments needed.



Why I Built This

Mostly because I wanted something I could:

Run before race week

Not have to constantly update

Actually reflect current form instead of vibes

It’s also a base I can keep improving over time (track-specific models, weather, Monte Carlo sims, etc.).
