# formula1
Formula 1 P1 Prediction 

1. Data Collection
- Uses FastF1 API to fetch qualifying session data
- Collects data from recent 2025 races (rounds 1-4)
- Includes 2024 Japanese GP data as reference

2. Data Processing
- Converts lap times from timedelta to seconds
- Handles missing values using SimpleImputer
- Cleans and structures data for analysis

3. Model Development
- Uses Linear Regression to establish baseline predictions
- Features: Q1 and Q2 times
- Target: Q3 times
- Includes train-test split for validation

4. Performance Factors
- Implements team-specific performance coefficients
- Adds driver-specific performance adjustments
- Base lap time calibrated to ~89.5 seconds
- Includes small random variation for realism

5. Prediction System
- Combines model predictions with performance factors
- Accounts for 2025 driver-team combinations
- Sorts and displays predicted qualifying order

6. Validation
- Calculates Mean Absolute Error (MAE)
- Provides R² score for model accuracy
- Visualizes qualifying time distributions
