import pandas as pd
import sys

def increment_csv_values(input_file, output_file, increment=2):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Increment values in numeric columns
        for col in numeric_columns:
            df[col] = df[col] + increment
        
        # Save the modified dataframe to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {input_file} and saved results to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    
    input_file = 'data/Cleanup_7/CF/cf_01.csv'  # Replace with your input file name
    output_file = 'data/Cleanup_7/CF/cf_01.csv'
    increment = 40 if len(sys.argv) < 4 else float(sys.argv[3])
    
    increment_csv_values(input_file, output_file, increment)
