import os
import pandas as pd
from generations import PATH as GENERATIONS_PATH


def aggregate_csvs(source_dir, output_file):
    # List all CSV files in the source directory
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv') and f != os.path.basename(output_file)]

    # Read and concatenate all CSV files
    combined_df = pd.concat(
        [pd.read_csv(os.path.join(source_dir, file)) for file in csv_files],
        ignore_index=True
    )

    # Save the combined DataFrame to the output file
    combined_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    output_csv = os.path.join(GENERATIONS_PATH, "all_responses.csv")
    # Aggregate all CSVs into one
    aggregate_csvs(GENERATIONS_PATH, output_csv)
