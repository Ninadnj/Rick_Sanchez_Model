from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Prarabdha/Rick_and_Morty_Transcript")
df = dataset["train"].to_pandas()

print("Columns:", df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10))

# Check if rows are sequential
print("\nChecking for dialogue continuity...")
for i in range(5):
    print(f"Row {i}: {df.iloc[i].get('speaker', 'Unknown')} - {df.iloc[i].get('line', df.iloc[i].get('dialouge', 'Unknown'))}")
