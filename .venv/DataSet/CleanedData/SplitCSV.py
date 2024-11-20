import csv

# Input file path
input_file = "cleaned_dataset.csv"
rows_per_chunk = 50000  # Number of rows per split file

# Split the CSV
with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read the header row

    file_count = 0
    rows = []

    for i, row in enumerate(reader, start=1):
        rows.append(row)
        if i % rows_per_chunk == 0:
            output_file = f"split_file_{file_count}.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                if file_count == 0:
                    writer.writerow(header)  # Write header to each split file
                writer.writerows(rows)
            rows = []
            file_count += 1

    # Write any remaining rows
    if rows:
        output_file = f"split_file_{file_count}.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(rows)
        file_count += 1

print(f"CSV file split into {file_count} files.")
