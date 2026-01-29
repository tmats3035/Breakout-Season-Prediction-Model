from pybaseball import batting_stats, statcast_batter_exitvelo_barrels
import pandas as pd
import joblib
import json

# Load the trained model
model = joblib.load('breakout_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# Get 2025 player data
season_stats_2025 = batting_stats(2025, 2025, qual=200)
statcast_2025 = statcast_batter_exitvelo_barrels(2025)
statcast_2025['year'] = 2025

# Process Statcast names
statcast_2025[['last_name', 'first_name']] = statcast_2025['last_name, first_name'].str.split(', ', expand=True)
statcast_2025['name_match'] = statcast_2025['first_name'] + ' ' + statcast_2025['last_name']

# Merge datasets
df_2025 = season_stats_2025.merge(
    statcast_2025, 
    left_on=['Name', 'Season'], 
    right_on=['name_match', 'year'], 
    how='inner'
)

# Engineer features (same as training)
df_2025['exit_velo_consistency'] = df_2025['max_hit_speed'] - df_2025['avg_hit_speed']
df_2025['power_flyball'] = (df_2025['avg_hit_speed'] * df_2025['FB%']) / 100
df_2025['barrel_vs_HR'] = (df_2025['brl_percent'] * 2) - df_2025['HR/FB']
df_2025['BB_K_ratio'] = df_2025['BB%'] / df_2025['K%'].replace(0, 0.01)
df_2025['is_young'] = (df_2025['Age'] <= 26).astype(int)
df_2025['is_part_time'] = (df_2025['PA'] < 400).astype(int)

# For 2025 players, we don't have year-over-year changes, so set them to 0
df_2025['K%_change'] = 0
df_2025['ISO_change'] = 0
df_2025['barrel_change'] = 0
df_2025['exit_velo_change'] = 0

# Select features and make predictions
X_2025 = df_2025[features].copy()
X_2025 = X_2025.fillna(0)  # Fill any missing values with 0

X_2025_scaled = scaler.transform(X_2025)
predictions = model.predict_proba(X_2025_scaled)[:, 1]

# Create output dataset
results = []
for idx, row in df_2025.iterrows():
    results.append({
        'name': row['Name'],
        'team': row['Team'],
        'age': int(row['Age']),
        'position': row['Pos'] if 'Pos' in row else 'Unknown',
        'war_2025': float(row['WAR']),
        'wrc_plus': int(row['wRC+']),
        'avg_exit_velo': float(row['avg_hit_speed']),
        'barrel_percent': float(row['brl_percent']),
        'breakout_probability': float(predictions[idx]),
        'prediction': 'Breakout' if predictions[idx] >= 0.5 else 'No Breakout'
    })

# Sort by breakout probability
results_sorted = sorted(results, key=lambda x: x['breakout_probability'], reverse=True)

# Save to JSON
with open('predictions_2025.json', 'w') as f:
    json.dump(results_sorted, f, indent=2)

print(f"Generated predictions for {len(results)} players")
print(f"Top 5 breakout candidates:")
for player in results_sorted[:5]:
    print(f"  {player['name']}: {player['breakout_probability']:.1%}")