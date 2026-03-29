#!/usr/bin/env python3
"""
Generate synthetic test data to replace real goalkeeper data.
This allows uploading to GitHub without sharing proprietary coaching data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Common goalkeeper first and last names (generic, not real players)
FIRST_NAMES = [
    "Alex", "Ben", "Chris", "David", "Erik", "Frank", "George", "Henry", 
    "Ivan", "Jack", "Kevin", "Lucas", "Martin", "Nathan", "Oscar", "Peter",
    "Quinn", "Robert", "Samuel", "Thomas", "Ulrich", "Victor", "Walter", "Xavier",
    "Yannick", "Zachary", "Adam", "Brian", "Daniel", "Edward", "Felix", "Gregory",
    "Harrison", "Isaac", "Justin", "Klaus", "Liam", "Michael", "Nicholas", "Oliver",
    "Patrick", "Quentin", "Raymond", "Stefan", "Trevor", "Ulysses", "Vincent", "William"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Peterson", "Phillips", "Campbell",
    "Roberts", "Rogers", "Morgan", "Peterson", "Cooper", "Reed", "Bell", "Gomez",
    "Murray", "Freeman", "Wells", "Webb", "Simpson", "Stevens", "Tucker", "Porter"
]

STATUSES = ["PLAYS", "BENCH", "STAYED", "DROPPED"]
DIRECTIONS = ["UP", "DOWN", "STAYED"]

def generate_player_id():
    """Generate random player ID"""
    return np.random.randint(10000, 200000)

def generate_player_name():
    """Generate random player name"""
    return f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"

def generate_features(n_samples):
    """Generate synthetic performance features"""
    features = {
        'mean_GK_PREVENTED_GOALS_TOTAL_POSTSHOT_XG': np.random.normal(0.1, 0.3, n_samples),
        'mean_GK_PREVENTED_GOALS_POST_SHOT_XG_BY_ACTION_LONG_RANGE_SHOT_RATIO': np.random.normal(0.02, 0.08, n_samples),
        'mean_GK_PREVENTED_GOALS_POST_SHOT_XG_BY_ACTION_MID_RANGE_SHOT_RATIO': np.random.normal(0.1, 0.15, n_samples),
        'mean_GK_PREVENTED_GOALS_POST_SHOT_XG_BY_ACTION_CLOSE_RANGE_SHOT_RATIO': np.random.normal(0.05, 0.12, n_samples),
        'mean_GK_PREVENTED_GOALS_POST_SHOT_XG_BY_ACTION1V1_AGAINST_GK_SHOT_RATIO': np.random.normal(0.03, 0.10, n_samples),
        'mean_GK_PREVENTED_GOALS_POST_SHOT_XG_BY_ACTION_HEADER_SHOT_RATIO': np.random.normal(0.05, 0.12, n_samples),
        'mean_GK_DEFENSIVE_TOUCHES_OUTSIDE_OWN_BOX': np.random.normal(0.8, 0.6, n_samples),
        'mean_GK_CAUGHT_AND_PUNCHED_HIGH_BALLS_PERCENT': np.random.uniform(0.05, 0.25, n_samples),
        'mean_GK_SUCCESSFUL_LAUNCHES_PERCENT': np.random.uniform(0.25, 0.65, n_samples),
        'mean_GOAL_KICK_SCORE': np.random.uniform(0.1, 0.25, n_samples),
        'mean_LOW_PASS_SCORE': np.random.uniform(0.08, 0.22, n_samples),
        'mean_DIAGONAL_PASS_SCORE': np.random.uniform(0.05, 0.35, n_samples),
        'mean_RATIO_PASSING_ACCURACY': np.random.uniform(0.65, 0.95, n_samples),
        'mean_PASS_COMPLETION_OVER_EXPECTED': np.random.uniform(-0.15, 0.15, n_samples),
        'mean_IMPECT_SCORE_PACKING': np.random.uniform(0.2, 0.5, n_samples),
        'mean_DEFENSIVE_IMPECT_SCORE_PACKING': np.random.uniform(-0.15, 0.15, n_samples),
        'mean_TOTAL_TOUCHES': np.random.uniform(15, 45, n_samples),
    }
    return pd.DataFrame(features)

def generate_gk_features_up(n_keepers=218):
    """Generate GK features UP dataset"""
    data = {
        'playerId': [generate_player_id() for _ in range(n_keepers)],
        'name': [generate_player_name() for _ in range(n_keepers)],
        'status': np.random.choice(STATUSES, n_keepers),
        'direction': np.random.choice(DIRECTIONS, n_keepers),
        'y_plays': np.random.randint(0, 2, n_keepers),
        'y_up': np.random.randint(0, 2, n_keepers),
        'age': np.random.uniform(19, 40, n_keepers),
        'origin_median': np.random.uniform(0.3, 0.8, n_keepers),
        'n_matches_loaded': np.random.randint(3, 100, n_keepers),
    }
    
    df = pd.DataFrame(data)
    features = generate_features(n_keepers)
    df = pd.concat([df, features], axis=1)
    
    return df

def generate_gk_features_ready(n_keepers=207):
    """Generate GK features READY dataset (slightly different from UP)"""
    data = {
        'playerId': [generate_player_id() for _ in range(n_keepers)],
        'name': [generate_player_name() for _ in range(n_keepers)],
        'status': np.random.choice(STATUSES, n_keepers),
        'age': np.random.uniform(19, 40, n_keepers),
        'origin_median': np.random.uniform(0.3, 0.8, n_keepers),
        'n_matches_loaded': np.random.randint(3, 100, n_keepers),
        'y_ready': np.random.randint(0, 2, n_keepers),
    }
    
    df = pd.DataFrame(data)
    features = generate_features(n_keepers)
    df = pd.concat([df, features], axis=1)
    
    return df

def generate_gk_features_plays(n_keepers=212):
    """Generate GK features PLAYS dataset"""
    data = {
        'playerId': [generate_player_id() for _ in range(n_keepers)],
        'name': [generate_player_name() for _ in range(n_keepers)],
        'status': np.random.choice(STATUSES, n_keepers),
        'plays': np.random.randint(0, 2, n_keepers),
        'age': np.random.uniform(19, 40, n_keepers),
        'origin_median': np.random.uniform(0.3, 0.8, n_keepers),
        'n_matches_loaded': np.random.randint(3, 100, n_keepers),
    }
    
    df = pd.DataFrame(data)
    features = generate_features(n_keepers)
    df = pd.concat([df, features], axis=1)
    
    return df

def main():
    base_path = Path(__file__).parent
    
    print("🔄 Generating synthetic test data...")
    
    # Generate datasets
    gk_up = generate_gk_features_up(218)
    gk_ready = generate_gk_features_ready(207)
    gk_plays = generate_gk_features_plays(212)
    
    # Write CSV files with _sample suffix
    gk_up.to_csv(base_path / "gk_features_up_sample.csv", index=False)
    print(f"✓ Generated gk_features_up_sample.csv ({len(gk_up)} keepers)")
    
    gk_ready.to_csv(base_path / "gk_features_ready_sample.csv", index=False)
    print(f"✓ Generated gk_features_ready_sample.csv ({len(gk_ready)} keepers)")
    
    gk_plays.to_csv(base_path / "gk_features_plays_sample.csv", index=False)
    print(f"✓ Generated gk_features_plays_sample.csv ({len(gk_plays)} keepers)")
    
    # Also generate a larger synthetic dataset in Data/ if needed
    gk_dataset = pd.concat([gk_up, gk_ready, gk_plays], ignore_index=True)
    gk_dataset = gk_dataset.drop_duplicates(subset=['playerId'], keep='first')
    gk_dataset.to_csv(base_path / "Data" / "gk_dataset_final_sample.csv", index=False)
    print(f"✓ Generated Data/gk_dataset_final_sample.csv ({len(gk_dataset)} keepers)")
    
    print("\n✅ All synthetic data generated successfully!")
    print("   These are completely made-up goalkeeper names and stats.")
    print("   Safe to upload to GitHub without any data sharing concerns.")

if __name__ == "__main__":
    main()
