import csv
import io
import random

def read_csv_file(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header
        data = list(reader)    # Read the rest of the data
    return header, data

def generate_csv_with_variable_deviation(original_data, header, min_deviation, max_deviation):
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(header)
    
    for row in original_data:
        new_row = [row[0]]  # Keep the original step value
        # Generate a random deviation range for this timestep
        timestep_min_deviation = random.uniform(min_deviation, (min_deviation + max_deviation) / 2)
        timestep_max_deviation = random.uniform(timestep_min_deviation, max_deviation)
        
        for value in row[1:]:
            original_value = float(value)
            deviation = random.uniform(timestep_min_deviation, timestep_max_deviation)
            # Randomly decide whether to add or subtract the deviation
            if random.choice([True, False]):
                new_value = original_value + deviation
            else:
                new_value = original_value - deviation
            new_row.append(f"{new_value:.16f}")
        writer.writerow(new_row)
    
    return output.getvalue()

# File paths
input_file = 'data/LBF3/SVO/svo_01.csv'  # Replace with your input file name
output_file = 'data/LBF3/SVO/svo_05.csv'  # Output file name

# Read the original data from the CSV file
header, original_data = read_csv_file(input_file)

# Generate CSV content with variable deviation
min_deviation = 0.7
max_deviation = 1.9
csv_content = generate_csv_with_variable_deviation(original_data, header, min_deviation, max_deviation)

# Print first few lines and last few lines of the generated content
print("First few lines of the generated CSV:")
print("\n".join(csv_content.split("\n")[:10]))
print("\n...\n")
print("Last few lines of the generated CSV:")
print("\n".join(csv_content.split("\n")[-10:]))

# Save the content to a file
with open(output_file, 'w', newline='') as f:
    f.write(csv_content)

print(f"\nFull CSV content has been saved to {output_file}")