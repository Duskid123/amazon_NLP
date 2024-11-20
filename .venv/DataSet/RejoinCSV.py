import csv
import glob

# Output file path
output_file = "rejoined_file.csv"

# Get list of split files (adjust pattern if needed)
split_files = sorted(glob.glob("split_file_*.csv"))

# Rejoin the files
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = None

    for split_file in split_files:
        with open(split_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)

            if writer is None:
                writer = csv.writer(outfile)
                writer.writerow(header)  # Write header only once

            writer.writerows(reader)

print(f"Split files rejoined into {output_file}.")
