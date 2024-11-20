import json
import csv

# Input and output file paths
json_file = "Toys_and_Games_5.json"  # Replace with your JSON file path
csv_file = "Toys_and_Games_5.csv"  # Replace with your desired CSV file path

# Open the JSON file and process each object separately
with open(json_file, 'r') as jf, open(csv_file, 'w', newline='', encoding='utf-8') as cf:
    writer = None
    for line in jf:
        data = json.loads(line.strip())  # Load each JSON object
        if writer is None:
            writer = csv.DictWriter(cf, fieldnames=data.keys())
            writer.writeheader()
        writer.writerow(data)

print(f"JSON has been successfully converted to {csv_file}")