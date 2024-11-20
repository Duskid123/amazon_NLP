import csv
import glob

# Output file path
output_file = "rejoined_Clean_file.csv"

# Get list of split files (adjust pattern if needed)
split_files = sorted(glob.glob("split_file_*.csv"))

# Rejoin the files
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = None
    header_written = False  # Flag to track if header has been written

    for split_file in split_files:
        with open(split_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read the header

            if not header_written:
                writer = csv.writer(outfile)
                writer.writerow(header)  # Write header only once
                header_written = True  # Set flag to True

            writer.writerows(reader)  # Write the remaining rows

print(f"Split files rejoined into {output_file}.")
