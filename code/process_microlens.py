
import pandas as pd

# Read the original CSV file
df = pd.read_csv("../information/microlens_ori.csv")  # Change 'input.csv' to your actual file name

# Add empty 'tag' and 'description' columns
df['tag'] = ""
df['description'] = ""

# Write to a new CSV file
df.to_csv("../information/microlens.csv", index=False)  # Change 'output.csv' if needed

print("CSV file with empty columns added successfully.")


# Read the CSV file
df = pd.read_csv("../dataset/microlens_ori.csv")

# Rename and reorder columns
df = df.rename(columns={"user": "user_id", "item": "item_id"})[["item_id", "user_id", "timestamp"]]

# Write to a new file
df.to_csv("../dataset/microlens.csv", index=False)


# with open(jsonl_file, 'r') as infile:
    
# with open(../, 'w', newline='') as outfile:
#     writer = csv.writer(outfile)
#     writer.writerow(['item_id', 'description', 'title', 'tag'])  # Write header
#     for item_id, description in seen_items.items():
#         # breakpoint()
#         description = description.replace("\n", "\\n")
#         writer.writerow([item_id, description, "", ""])