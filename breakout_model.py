from pybaseball import batting_stats, statcast_batter_exitvelo_barrels
import pandas as pd

season_stats = batting_stats(2019, 2025, qual=200)
season_stats = season_stats[season_stats['Season'] != 2020]

statcast_2019 = statcast_batter_exitvelo_barrels(2019)
statcast_2021 = statcast_batter_exitvelo_barrels(2021)
statcast_2022 = statcast_batter_exitvelo_barrels(2022)
statcast_2023 = statcast_batter_exitvelo_barrels(2023)
statcast_2024 = statcast_batter_exitvelo_barrels(2024)
statcast_2025 = statcast_batter_exitvelo_barrels(2025)

statcast_all = pd.concat([statcast_2019, statcast_2021, statcast_2022, statcast_2023, statcast_2024, statcast_2025])

print(f"Season stats shape: {season_stats.shape}")
print(f"Statcast shape: {statcast_all.shape}")
print(f"Statcast columns: {statcast_all.columns.tolist()}")