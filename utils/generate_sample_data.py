import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_load_data(filename='data/historical_load.csv', days=30):
    """
    Generates synthetic electrical load data with daily and weekly patterns.
    """
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    # Base load (mean)
    base_load = 500 
    
    # Daily pattern (sine wave)
    daily_pattern = 100 * np.sin(np.linspace(0, 2 * np.pi * days, days * 24))
    
    # Weekly pattern (lower load on weekends)
    weekly_pattern = np.array([20 if (start_date + timedelta(hours=i)).weekday() < 5 else -30 for i in range(days * 24)])
    
    # Random noise
    noise = np.random.normal(0, 15, days * 24)
    
    load = base_load + daily_pattern + weekly_pattern + noise
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'load': load
    })
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Sample data generated: {filename}")

if __name__ == "__main__":
    generate_sample_load_data()
