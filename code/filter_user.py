import argparse
import pandas as pd
import hashlib
import math
def deterministic_sample(user_ids, fraction=50):
    """Deterministically pick half of the user_ids based on their hash values."""
    # breakpoint()
    fraction = 1 - int(fraction) / 100
    # sorted_users = sorted(user_ids, key=lambda x: hashlib.md5(x.encode()).hexdigest())
    sorted_users = sorted(user_ids)
    return set(sorted_users[:math.floor(len(sorted_users) * fraction)])

def filter_csv(input_file1, output_file1, input_file2, output_file2, percent):
    # Read the first input CSV
    df1 = pd.read_csv(input_file1)
    
    # Get unique users and sample them deterministically
    all_users = set(df1['user_id'].unique())
    sampled_users = deterministic_sample(all_users, fraction=percent)
    
    # Filter out sampled users
    filtered_df1 = df1[~df1['user_id'].isin(sampled_users)]
    filtered_df1.to_csv(output_file1, index=False)
    
    # Read the second input CSV
    df2 = pd.read_csv(input_file2)
    
    # Extract item_ids from the remaining user records
    remaining_item_ids = set(filtered_df1['item_id'].unique())
    
    # Filter df2 to include only rows with matching item_id
    filtered_df2 = df2[df2['item_id'].isin(remaining_item_ids)]
    
    # Write to the second output CSV
    filtered_df2.to_csv(output_file2, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input CSV file")
    parser.add_argument("--input_file2", required=True, help="Path to input CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save filtered CSV file")
    parser.add_argument("--output_file2", required=True, help="Path to save filtered CSV file")
    parser.add_argument("--percent",type=int, required=True, help="percentage, in the python code, it is divided by 100")
    args = parser.parse_args()
    
    filter_csv(args.input_file, args.output_file, args.input_file2, args.output_file2, args.percent)
