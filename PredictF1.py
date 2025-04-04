import os
import fastf1 as ff1
import pandas as pd
import matplotlib

# Ensure cache directory exists
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable caching
ff1.Cache.enable_cache(cache_dir)

session = ff1.get_session(2024, "Japan", "Race")
session.load()
laps = session.laps
# df = laps[['Driver', 'LapTime','Compound']]

print(laps.columns)
