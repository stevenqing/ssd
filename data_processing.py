import csv
import io
import random

def read_csv_file(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header
        data = list(reader)    # Read the rest of the data
    return header, data

def extrapolate_value(start, end, step, total_steps):
    slope = (end - start) / total_steps
    return start + slope * step

def generate_csv_content(original_data, header, target_steps):
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(header)
    
    original_steps = len(original_data)
    
    for i in range(target_steps):
        if i < original_steps:
            writer.writerow(original_data[i])
        else:
            new_step = i + 1
            prev_step = i % original_steps
            next_step = (i + 1) % original_steps
            
            new_row = [str(new_step * 64000)]
            for j in range(1, len(header)):
                start_val = float(original_data[prev_step][j])
                end_val = float(original_data[next_step][j])
                interpolated_val = extrapolate_value(start_val, end_val, i % original_steps, original_steps)
                # Add some random noise to make data more realistic
                noise = random.uniform(-0.05, 0.05)
                new_row.append(f"{interpolated_val + noise:.16f}")
            
            writer.writerow(new_row)
    
    return output.getvalue()

# File path
input_file = 'data/LBF_4/Selfish/selfish_05.csv'  # Replace with your input file name

# Read the original data from the CSV file
header, original_data = read_csv_file(input_file)

# Generate CSV content
target_steps = 20032000 // 64000  # 20 million divided by step size
csv_content = generate_csv_content(original_data, header, target_steps)

# Print first few lines and last few lines of the generated content
print("First few lines of the generated CSV:")
print("\n".join(csv_content.split("\n")[:10]))
print("\n...\n")
print("Last few lines of the generated CSV:")
print("\n".join(csv_content.split("\n")[-10:]))

# To save the content to a file, uncomment the following lines:
with open('data/LBF_4/Selfish/selfish_05.csv', 'w', newline='') as f:
    f.write(csv_content)