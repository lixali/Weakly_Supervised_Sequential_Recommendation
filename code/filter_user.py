import argparse
import pandas as pd
import hashlib
import math
def deterministic_sample(user_ids, fraction=0.90):
    """Deterministically pick half of the user_ids based on their hash values."""
    sorted_users = sorted(user_ids, key=lambda x: hashlib.md5(x.encode()).hexdigest())
    return set(sorted_users[:math.floor(len(sorted_users) * fraction)])

def filter_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    all_users = set(df['user_id'].unique())
    sampled_users = deterministic_sample(all_users)
    
    filtered_df = df[~df['user_id'].isin(sampled_users)]
    filtered_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save filtered CSV file")
    args = parser.parse_args()
    
    filter_csv(args.input_file, args.output_file)
