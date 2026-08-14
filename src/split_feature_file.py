import gzip
import csv
import os

INPUT_FILE = r"output\customer_product_features.csv.gz"
OUTPUT_DIR = r"output\feature_parts"

ROWS_PER_PART = 2_200_000


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting file split...")
print(f"Input: {INPUT_FILE}")
print(f"Rows per part: {ROWS_PER_PART:,}")
print()

with gzip.open(INPUT_FILE, "rt", encoding="utf-8", newline="") as infile:

    reader = csv.reader(infile)

    # Read header
    header = next(reader)

    part_number = 1
    row_count = 0

    output_file = None
    writer = None

    for row in reader:

        # Start a new output file
        if row_count % ROWS_PER_PART == 0:

            if output_file is not None:
                output_file.close()

            output_path = os.path.join(
                OUTPUT_DIR,
                f"customer_product_features_part_{part_number}.csv.gz"
            )

            output_file = gzip.open(
                output_path,
                "wt",
                encoding="utf-8",
                newline=""
            )

            writer = csv.writer(output_file)

            # Write header to every part
            writer.writerow(header)

            print(f"Created part {part_number}: {output_path}")

            part_number += 1

        writer.writerow(row)

        row_count += 1

        if row_count % 500_000 == 0:
            print(f"Processed {row_count:,} rows")

    if output_file is not None:
        output_file.close()


print()
print("======================================")
print("FILE SPLITTING COMPLETED")
print("======================================")
print(f"Total rows processed: {row_count:,}")
print(f"Parts created: {part_number - 1}")
print()
print(f"Output directory: {OUTPUT_DIR}")