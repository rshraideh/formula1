import fastf1
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# CONFIG (Set once, never touch)
# ==============================
N_PREVIOUS_SEASONS = 2      # How many past seasons to train on
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Enable FastF1 cache
fastf1.Cache.enable_cache("cache")

# ==============================
# DATA FETCHING
# ==============================
def fetch_historical_data(season_year):
    schedule = fastf1.get_event_schedule(season_year)
    now_utc = pd.Timestamp.now(tz="UTC")

    completed_events = schedule[schedule["Session5DateUtc"] < now_utc]
    all_sessions = []

    for _, event in completed_events.iterrows():
        try:
            rnd = int(event["RoundNumber"])
            quali = fastf1.get_session(season_year, rnd, "Q")
            quali.load()

            df = quali.results[
                ["DriverNumber", "FullName", "TeamName", "Q1", "Q2", "Q3"]
            ].rename(columns={"FullName": "Driver"})

            for col in ["Q1", "Q2", "Q3"]:
                df[f"{col}_sec"] = df[col].apply(
                    lambda x: x.total_seconds() if pd.notnull(x) else np.nan
                )

            df["Year"] = season_year
            df["Round"] = rnd
            all_sessions.append(df)

        except Exception:
            continue

    return pd.concat(all_sessions, ignore_index=True) if all_sessions else pd.DataFrame()


def get_relevant_seasons(n_previous=2):
    current_year = datetime.datetime.utcnow().year
    return [current_year - i - 1 for i in range(n_previous)]


# ==============================
# MODEL TRAINING
# ==============================
def train_qualifying_model(df):
    df = df.dropna(subset=["Q1_sec", "Q2_sec", "Q3_sec"])

    X = df[["Q1_sec", "Q2_sec"]]
    y = df["Q3_sec"]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    latest_round = df["Round"].max()
    weights = 1 / (latest_round - df["Round"] + 1)

    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)

    preds = model.predict(X)
    print(
        f"Model trained | MAE: {mean_absolute_error(y, preds):.3f}s | "
        f"R²: {r2_score(y, preds):.3f}"
    )

    return model, df


# ==============================
# NEXT RACE DETECTION
# ==============================
def get_next_race(season_year):
    schedule = fastf1.get_event_schedule(season_year)
    now_utc = pd.Timestamp.now(tz="UTC")
    future_events = schedule[schedule["Session5DateUtc"] > now_utc]

    if future_events.empty:
        return None

    next_event = future_events.sort_values("Session5DateUtc").iloc[0]
    return {
        "round": int(next_event["RoundNumber"]),
        "name": next_event["OfficialEventName"],
        "date": next_event["EventDate"].date(),
    }


# ==============================
# DRIVER / TEAM DETECTION
# ==============================
def get_current_grid(training_df):
    return (
        training_df[["Driver", "TeamName"]]
        .drop_duplicates()
        .rename(columns={"TeamName": "Team"})
    )


# ==============================
# PREDICTION ENGINE
# ==============================
def predict_qualifying_and_race(training_df, grid_df):
    base_time = training_df["Q3_sec"].median()

    team_factor = (
        training_df.groupby("TeamName")["Q3_sec"].median() / base_time
    ).to_dict()

    driver_factor = (
        training_df.groupby("Driver")["Q3_sec"].median() / base_time
    ).to_dict()

    preds = grid_df.copy()
    preds["TeamFactor"] = preds["Team"].map(team_factor).fillna(1.0)
    preds["DriverFactor"] = preds["Driver"].map(driver_factor).fillna(1.0)

    preds["Predicted_Q3"] = (
        base_time * preds["TeamFactor"] * preds["DriverFactor"]
        + np.random.normal(0, 0.05, len(preds))
    )

    preds = preds.sort_values("Predicted_Q3").reset_index(drop=True)
    preds["Predicted_Race_Position"] = preds.index + 1

    return preds


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("Initializing fully automated F1 prediction model...\n")

    historical_dfs = []
    for season in get_relevant_seasons(N_PREVIOUS_SEASONS):
        print(f"Loading season {season}...")
        df = fetch_historical_data(season)
        if not df.empty:
            historical_dfs.append(df)

    if not historical_dfs:
        raise RuntimeError("No historical data available")

    combined_df = pd.concat(historical_dfs, ignore_index=True)

    model, training_df = train_qualifying_model(combined_df)

    current_season = datetime.datetime.utcnow().year
    next_race = get_next_race(current_season)

    if not next_race:
        print("No upcoming race detected.")
        exit()

    print(
        f"\nNext Grand Prix: {next_race['name']} "
        f"(Round {next_race['round']} – {next_race['date']})"
    )

    grid = get_current_grid(training_df)
    predictions = predict_qualifying_and_race(training_df, grid)

    print("\n=== QUALIFYING PREDICTION (Top 10) ===")
    for i, row in predictions.head(10).iterrows():
        print(
            f"{i+1:>2}. {row['Driver']:<20} "
            f"{row['Team']:<18} "
            f"{row['Predicted_Q3']:.3f}s"
        )

    print("\n=== RACE FINISH PREDICTION ===")
    for i, row in predictions.iterrows():
        print(
            f"{row['Predicted_Race_Position']:>2}. "
            f"{row['Driver']:<20} {row['Team']:<18}"
        )
