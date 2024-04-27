import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data/en_songs.csv')

# Determine the number of rows in the DataFrame
total_rows = len(df)

# Calculate the size of each part
part_size = total_rows // 20
remainder = total_rows % 20

# Split the DataFrame into 5 equal parts
start_idx = 0
for part_num in range(20):
    # Calculate the end index for the current part
    end_idx = start_idx + part_size
    if part_num < remainder:
        end_idx += 1  # Add one extra row to the first 'remainder' parts

    # Extract the current part from the DataFrame
    part_df = df.iloc[start_idx:end_idx]

    # Save the current part to a CSV file
    part_df.to_csv(f'part_{part_num + 1}.csv', index=False)

    # Update the start index for the next part
    start_idx = end_idx

print("Data split into 20 equal parts and saved successfully!")