import os
import fastf1 as ff1
import pandas as pd
import numpy as np
import pickle
import gzip
import matplotlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

# Ensure cache directory exists
def fetch_data(year, country):
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # # Enable caching
    ff1.Cache.enable_cache(cache_dir)

    session = ff1.get_session(year, country, "Race")
    session.load()

    df = session.results[['DriverNumber','BroadcastName','TeamName','Position','GridPosition','Time','Status']]

    df.to_csv(f'{year}_{country}_GP.csv', index=False)

    return df

#input Q1-Q3 values
def update_value(updated_data, filename):
    updates_df = pd.DataFrame(updated_data, columns=["BroadcastName", "Q1", "Q2", "Q3"])
    df= pd.read_csv(filename)
    df = df.merge(updates_df, on="BroadcastName", how="left")
    df.to_csv(filename, index=False)

def update_time_column(df):
    # Convert the Time column from string format to timedelta, handling NaN values
    time_deltas = pd.to_timedelta(df['Time'], errors='coerce')
    
    # Calculate the cumulative time using cumsum()
    cumulative_times = time_deltas.cumsum()
    
    # Convert the cumulative times to formatted strings, setting NaN values to "00:00.000"
    def format_timedelta(td):
        if pd.isna(td):
            return "00:00.000000"
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    # Apply the formatting function
    df['Time'] = cumulative_times.apply(format_timedelta)
    
    return df

japan_2022_data = [
    ["M VERSTAPPEN", "1:30.224", "1:30.346", "1:29.304"],
    ["C LECLERC", "1:30.402", "1:30.486", "1:29.314"],
    ["C SAINZ", "1:30.336", "1:30.444", "1:29.361"],
    ["S PEREZ", "1:30.622", "1:29.925", "1:29.709"],
    ["E OCON", "1:30.696", "1:30.357", "1:30.165"],
    ["L HAMILTON", "1:30.906", "1:30.443", "1:30.261"],
    ["F ALONSO", "1:30.603", "1:30.343", "1:30.322"],
    ["G RUSSELL", "1:30.865", "1:30.465", "1:30.389"],
    ["S VETTEL", "1:31.256", "1:30.656", "1:30.554"],
    ["L NORRIS", "1:30.881", "1:30.473", "1:31.003"],
    ["D RICCIARDO", "1:30.880", "1:30.659", "0:00.000"],
    ["V BOTTAS", "1:31.226", "1:30.709", "0:00.000"],
    ["Y TSUNODA", "1:31.130", "1:30.808", "0:00.000"],
    ["G ZHOU", "1:30.894", "1:30.953", "0:00.000"],
    ["M SCHUMACHER", "1:31.152", "1:31.439", "0:00.000"],
    ["A ALBON", "1:31.311", "0:00.000", "0:00.000"],
    ["P GASLY", "1:31.322", "0:00.000", "0:00.000"],
    ["K MAGNUSSEN", "1:31.352", "0:00.000", "0:00.000"],
    ["L STROLL", "1:31.419", "0:00.000", "0:00.000"],
    ["N LATIFI", "1:31.511", "0:00.000", "0:00.000"],
]

japan_2024_data = [
    ["M VERSTAPPEN", "1:28.866", "1:28.740", "1:28.197"],
    ["S PEREZ", "1:29.303", "1:28.752", "1:28.263"],
    ["L NORRIS", "1:29.536", "1:28.940", "1:28.489"],
    ["C SAINZ", "1:29.513", "1:29.099", "1:28.682"],
    ["F ALONSO", "1:29.254", "1:29.082", "1:28.686"],
    ["O PIASTRI", "1:29.425", "1:29.148", "1:28.760"],
    ["L HAMILTON", "1:29.661", "1:28.887", "1:28.766"],
    ["C LECLERC", "1:29.338", "1:29.196", "1:28.786"],
    ["G RUSSELL", "1:29.799", "1:29.140", "1:29.008"],
    ["Y TSUNODA", "1:29.775", "1:29.417", "1:29.413"],
    ["D RICCIARDO", "1:29.727", "1:29.472", "0:00.000"],
    ["N HULKENBERG", "1:29.821", "1:29.494", "0:00.000"],
    ["V BOTTAS", "1:29.602", "1:29.593", "0:00.000"],
    ["A ALBON", "1:29.963", "1:29.714", "0:00.000"],
    ["E OCON", "1:29.811", "1:29.816", "0:00.000"],
    ["L STROLL", "1:30.024", "0:00.000", "0:00.000"],
    ["P GASLY", "1:30.119", "0:00.000", "0:00.000"],
    ["K MAGNUSSEN", "1:30.131", "0:00.000", "0:00.000"],
    ["L SARGEANT", "1:30.139", "0:00.000", "0:00.000"],
    ["G ZHOU", "1:30.143", "0:00.000", "0:00.000"],
]

japan_2023_data = [
    ["M VERSTAPPEN", "1:29.878", "1:29.964", "1:28.877"],
    ["O PIASTRI", "1:30.439", "1:30.122", "1:29.458"],
    ["L NORRIS", "1:30.063", "1:30.296", "1:29.493"],
    ["C LECLERC", "1:30.393", "1:29.940", "1:29.542"],
    ["S PEREZ", "1:30.652", "1:29.965", "1:29.650"],
    ["C SAINZ", "1:30.651", "1:30.067", "1:29.850"],
    ["L HAMILTON", "1:30.811", "1:30.040", "1:29.908"],
    ["G RUSSELL", "1:30.811", "1:30.268", "1:30.219"],
    ["Y TSUNODA", "1:30.733", "1:30.204", "1:30.303"],
    ["F ALONSO", "1:30.971", "1:30.465", "1:30.560"],
    ["L LAWSON", "1:30.425", "1:30.508", "0:00.000"],
    ["P GASLY", "1:30.843", "1:30.509", "0:00.000"],
    ["A ALBON", "1:30.941", "1:30.537", "0:00.000"],
    ["E OCON", "1:30.960", "1:30.586", "0:00.000"],
    ["K MAGNUSSEN", "1:30.976", "1:30.665", "0:00.000"],
    ["V BOTTAS", "1:31.049", "0:00.000", "0:00.000"],
    ["L STROLL", "1:31.181", "0:00.000", "0:00.000"],
    ["N HULKENBERG", "1:31.299", "0:00.000", "0:00.000"],
    ["G ZHOU", "1:31.398", "0:00.000", "0:00.000"],
    ["L SARGEANT", "DNF", "0:00.000", "0:00.000"]
]

japan_2025_data = [
    ["M VERSTAPPEN", "Red Bull Racing", "1:27.943", "1:27.502", "1:26.983","1"],
    ["L NORRIS", "McLaren", "1:27.845", "1:27.146", "1:26.995","2"],
    ["O PIASTRI", "McLaren", "1:27.687", "1:27.507", "1:27.027","3"],
    ["C LECLERC", "Ferrari", "1:27.920", "1:27.555", "1:27.299","4"],
    ["G RUSSELL", "Mercedes", "1:27.843", "1:27.400", "1:27.318","5"],
    ["K ANTONELLI", "Mercedes", "1:27.968", "1:27.639", "1:27.555","6"],
    ["I HADJAR", "RB", "1:28.278", "1:27.775", "1:27.569","7"],
    ["L HAMILTON", "Ferrari", "1:27.942", "1:27.610", "1:27.610","8"],
    ["A ALBON", "Williams", "1:28.218", "1:27.783", "1:27.615","9"],
    ["O BEARMAN", "Haas F1 Team", "1:28.228", "1:27.711", "1:27.867","10"],
    ["P GASLY", "Alpine", "1:28.186", "1:27.822", "0:00.000","11"],
    ["C SAINZ", "Williams", "1:28.209", "1:27.836", "0:00.000","12"],
    ["F ALONSO", "Aston Martin", "1:28.337", "1:27.897", "0:00.000","13"],
    ["L LAWSON", "RB", "1:28.554", "1:27.906", "0:00.000","14"],
    ["Y TSUNODA", "Red Bull Racing", "1:27.967", "1:28.000", "0:00.000","15"],
    ["N HULKENBERG", "Kick Sauber", "1:28.570", "0:00.000", "0:00.000","16"],
    ["G BORTOLETO", "Kick Sauber", "1:28.622", "0:00.000", "0:00.000","17"],
    ["E OCON", "Haas F1 Team", "1:28.696", "0:00.000", "0:00.000","18"],
    ["J DOOHAN", "Alpine", "1:28.877", "0:00.000", "0:00.000","19"],
    ["L STROLL", "Aston Martin", "1:29.271", "0:00.000", "0:00.000","20"]
]

def process_2025_data(data):
    """Convert 2025 qualifying data into a structured DataFrame."""
    df = pd.DataFrame(data, columns=["BroadcastName", "TeamName", "Q1", "Q2", "Q3", "GridPosition"])
    df['Year'] = 2025
    
    # Convert qualifying times to seconds
    for q in ['Q1', 'Q2', 'Q3']:
        df[q] = df[q].apply(lambda x: f'00:{x}' if isinstance(x, str) and ':' in x and x.count(':') == 1 else x)
        df[f'{q}_seconds'] = pd.to_timedelta(df[q], errors='coerce').dt.total_seconds()
        df[f'{q}_seconds'] = df[f'{q}_seconds'].replace(0, np.nan)
    
    df['Best_Quali_Time'] = df[['Q1_seconds', 'Q2_seconds', 'Q3_seconds']].min(axis=1)
    df['GridPosition'] = pd.to_numeric(df['GridPosition'])
    
    return df

# --- LOAD HISTORICAL DATA ---
def load_historical_data(files_dict):
    """Load and preprocess historical race data."""
    all_data = []
    
    for year, filepath in files_dict.items():
        df = pd.read_csv(filepath)
        df['Year'] = year
        
        # Convert race times
        df['TimeSeconds'] = pd.to_timedelta(df['Time'], errors='coerce').dt.total_seconds()
        df['Finished'] = ~((df['Status'] == 'Retired') | (df['TimeSeconds'] == 0))
        
        # Convert qualifying times
        for q in ['Q1', 'Q2', 'Q3']:
            df[q] = df[q].apply(lambda x: f'00:{x}' if isinstance(x, str) and ':' in x and x.count(':') == 1 else x)
            df[f'{q}_seconds'] = pd.to_timedelta(df[q], errors='coerce').dt.total_seconds()
            df[f'{q}_seconds'] = df[f'{q}_seconds'].replace(0, np.nan)
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

# --- FEATURE ENGINEERING (WITH TEAM CHANGES) ---
def engineer_features(data):
    """Engineer features considering team changes."""
    data = data.copy()
    
    # Basic features
    data['Position_Delta'] = data['GridPosition'] - data['Position']
    data['Best_Quali_Time'] = data[['Q1_seconds', 'Q2_seconds', 'Q3_seconds']].min(axis=1)
    
    # Team strength (average position per team per year)
    team_strength = data.groupby(['Year', 'TeamName'])['Position'].mean().reset_index()
    team_strength.columns = ['Year', 'TeamName', 'Team_Strength']
    data = pd.merge(data, team_strength, on=['Year', 'TeamName'])
    
    # Driver strength relative to their team
    driver_performance = data.groupby(['Year', 'BroadcastName', 'TeamName'])['Position'].mean().reset_index()
    driver_performance.columns = ['Year', 'BroadcastName', 'TeamName', 'Driver_Position']
    
    # Calculate how much better/worse a driver is than their team's average
    data = pd.merge(data, driver_performance, on=['Year', 'BroadcastName', 'TeamName'])
    data['Driver_Relative_Strength'] = data['Team_Strength'] - data['Driver_Position']
    
    return data

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    """Evaluate model performance using cross-validation."""
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate absolute errors
    absolute_errors = np.abs(y_test - y_pred)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    winner_accuracy = (y_test.iloc[np.argmin(y_pred)] == y_test.min())
    podium_accuracy = np.mean(np.isin(y_pred.argsort()[:3], y_test.argsort()[:3]))
    
    return {
        'MAE': mae,
        'Winner_Accuracy': winner_accuracy,
        'Podium_Accuracy': podium_accuracy,
        'Absolute_Errors': absolute_errors
    }

def cross_validate_by_year_rfr(data, features, target='Position'):
    """Leave-one-year-out cross-validation."""
    years = sorted(data['Year'].unique())
    results = {}
    
    for test_year in years:
        # Split data
        train_data = data[data['Year'] != test_year]
        test_data = data[data['Year'] == test_year]
        
        X_train = train_data[features].fillna(0)
        y_train = train_data[target]
        X_test = test_data[features].fillna(0)
        y_test = test_data[target]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        performance = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)
        results[test_year] = performance
    
    return results

def cross_validate_by_year_xgb(data, features, target='Position'):
    """Leave-one-year-out cross-validation using XGBoost."""
    years = sorted(data['Year'].unique())
    results = {}

    for test_year in years:
        # Split data
        train_data = data[data['Year'] != test_year]
        test_data = data[data['Year'] == test_year]
        
        X_train = train_data[features].fillna(0)
        y_train = train_data[target]
        X_test = test_data[features].fillna(0)
        y_test = test_data[target]

        # Scale features and keep them as DataFrames
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

        # Train XGBoost model
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate model performance
        performance = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, scaler)
        results[test_year] = performance

    return results

def print_accuracy_metrics(validation_results):
    """Print accuracy metrics."""
    print("\nModel Accuracy Evaluation")
    print("-------------------------")
    for year, metrics in validation_results.items():
        print(f"Year {year}:")
        print(f"  - MAE: {metrics['MAE']:.2f} positions")
        print(f"  - Correctly Predicted Winner: {'Yes' if metrics['Winner_Accuracy'] else 'No'}")
        print(f"  - Podium Accuracy: {metrics['Podium_Accuracy']:.0%}")
        print(f"  - Max Prediction Error: {metrics['Absolute_Errors'].max():.1f} positions")
    
    avg_mae = np.mean([metrics['MAE'] for metrics in validation_results.values()])
    winner_accuracy = np.mean([metrics['Winner_Accuracy'] for metrics in validation_results.values()])
    podium_accuracy = np.mean([metrics['Podium_Accuracy'] for metrics in validation_results.values()])
    
    print("\nSummary Statistics")
    print(f"Average MAE: {avg_mae:.2f} positions")
    print(f"Winner Prediction Accuracy: {winner_accuracy:.0%}")
    print(f"Podium Prediction Accuracy: {podium_accuracy:.0%}")

def predict_2025_results_rfr(historical_data, current_data):
    """Predict 2025 results considering team changes."""
    # Train model on historical data
    features = ['GridPosition', 'Best_Quali_Time', 'Team_Strength', 'Driver_Relative_Strength']
    X = historical_data[features].fillna(0)
    y = historical_data['Position']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Prepare 2025 data
    latest_year = historical_data['Year'].max()
    latest_team_strength = historical_data[historical_data['Year'] == latest_year].groupby('TeamName')['Team_Strength'].mean()
    
    # Assign team strengths (use latest available or average for new teams)
    current_data['Team_Strength'] = current_data['TeamName'].map(latest_team_strength).fillna(latest_team_strength.mean())
    
    # For drivers with historical data, use their relative performance
    driver_history = historical_data.groupby('BroadcastName')['Driver_Relative_Strength'].mean().reset_index()
    current_data = pd.merge(current_data, driver_history, on='BroadcastName', how='left')
    
    # New drivers get average relative performance
    current_data['Driver_Relative_Strength'] = current_data['Driver_Relative_Strength'].fillna(0)
    
    # Make predictions
    X_current = current_data[features].fillna(0)
    X_current_scaled = scaler.transform(X_current)
    current_data['Predicted_Position'] = model.predict(X_current_scaled)
    
    return current_data.sort_values('Predicted_Position')

def predict_2025_results_xgb(historical_data, current_data):
    """Predict 2025 results using XGBoost considering team changes."""
    # Define features and target
    features = ['GridPosition', 'Best_Quali_Time', 'Team_Strength', 'Driver_Relative_Strength']
    X = historical_data[features].fillna(0)
    y = historical_data['Position']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    
    # Extract latest team strength from last year
    latest_year = historical_data['Year'].max()
    latest_team_strength = historical_data[historical_data['Year'] == latest_year].groupby('TeamName')['Team_Strength'].mean()
    
    # Assign team strengths to 2025 data
    current_data['Team_Strength'] = current_data['TeamName'].map(latest_team_strength).fillna(latest_team_strength.mean())
    
    # Assign driver relative strength
    driver_history = historical_data.groupby('BroadcastName')['Driver_Relative_Strength'].mean().reset_index()
    current_data = pd.merge(current_data, driver_history, on='BroadcastName', how='left')
    current_data['Driver_Relative_Strength'] = current_data['Driver_Relative_Strength'].fillna(0)
    
    # Scale and predict
    X_current = current_data[features].fillna(0)
    X_current_scaled = scaler.transform(X_current)
    current_data['Predicted_Position'] = model.predict(X_current_scaled)
    
    return current_data.sort_values('Predicted_Position')

# def visualize_results(predicted_results):
#     """Plot predicted 2025 results."""
#     plt.figure(figsize=(12, 8))
#     top_results = predicted_results.head(10)
#     sns.barplot(x='Predicted_Position', y='BroadcastName', data=top_results)
#     plt.title('Predicted Top 10 Finishers - 2025 Japanese GP')
#     plt.xlabel('Predicted Position')
#     plt.ylabel('Driver')
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":

    # Load and preprocess data
    data_files = {2022: "2022_Japan_GP.csv", 2023: "2023_Japan_GP.csv", 2024: "2024_Japanese_GP.csv"}
    historical_data = load_historical_data(data_files)
    processed_data = engineer_features(historical_data)
    
    # Define features
    features = ['GridPosition', 'Best_Quali_Time', 'Team_Strength', 'Driver_Relative_Strength']
    
    # Cross-validate
    rfr_validation_results = cross_validate_by_year_rfr(processed_data, features)
    xgb_validation_results = cross_validate_by_year_xgb(processed_data, features)
    print("RFR:\n")
    print_accuracy_metrics(rfr_validation_results)
    print("XGB:\n")
    print_accuracy_metrics(xgb_validation_results)
    
    
    # Predict 2025 results
    current_data = process_2025_data(japan_2025_data)
    rfr_predicted_results = predict_2025_results_rfr(processed_data, current_data)
    
    # Display predictions
    print("\nPredicted 2025 Japanese GP Results")
    print("----------------------------------")
    print(rfr_predicted_results[['BroadcastName', 'TeamName', 'GridPosition', 'Predicted_Position']].head(10).to_string(index=False))

    xgb_predicted_results = predict_2025_results_xgb(processed_data, current_data)
    print("\nPredicted 2025 Japanese GP Results")
    print("----------------------------------")
    print(xgb_predicted_results[['BroadcastName', 'TeamName', 'GridPosition', 'Predicted_Position']].head(10).to_string(index=False))