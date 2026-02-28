import csv
import sys
import os

def analyze_performance(filename):
    """
    Reads a performance CSV, finds peak GFLOPS, and identifies occupancy 
    values for runs achieving > 90% of that peak.

    Args:
        filename (str): The path to the CSV file.
    """
    
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        print("Please check the file path and try again.")
        return

    data = []
    max_gflops = 0.0

    # --- First Pass: Read data and find max GFLOPS ---
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            # Check for empty file
            first_char = file.read(1)
            if not first_char:
                print(f"Error: The file '{filename}' is empty.")
                return
            file.seek(0) # Reset file pointer

            reader = csv.DictReader(file)
            
            # Check for expected headers
            expected_headers = ['GFLOPS', 'MaxAttainedOccupancy']
            if not all(h in reader.fieldnames for h in expected_headers):
                print(f"Error: CSV missing required headers. Found: {reader.fieldnames}")
                print(f"Expected at least: {expected_headers}")
                return

            for i, row in enumerate(reader):
                try:
                    gflops = float(row['GFLOPS'])
                    occupancy = int(row['MaxAttainedOccupancy'])
                    
                    data.append({'GFLOPS': gflops, 'Occupancy': occupancy})
                    
                    if gflops > max_gflops:
                        max_gflops = gflops
                        
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping invalid data at row {i+2}: {row}. Error: {e}")
                except KeyError as e:
                    print(f"Warning: Skipping row {i+2} due to missing key: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    # --- Analysis ---
    if not data:
        print("No valid data was read from the file.")
        return

    # Calculate the 90% performance threshold
    performance_threshold = max_gflops * 0.90

    # --- Second Pass: Filter data based on the threshold ---
    high_perf_occupancies = set()
    for row in data:
        if row['GFLOPS'] > performance_threshold:
            high_perf_occupancies.add(row['Occupancy'])

    # --- Print Results ---
    print("--- Performance Analysis ---")
    print(f"Peak Performance (Max GFLOPS): {max_gflops:.4f}")
    print(f"90% Performance Threshold: {performance_threshold:.4f} GFLOPS")
    print("-" * 30)

    if not high_perf_occupancies:
        print("No experiments were found that exceeded the 90% performance threshold.")
    else:
        print("Found Occupancy values for experiments > 90% peak performance:")
        # Sort the set for consistent output
        sorted_occupancies = sorted(list(high_perf_occupancies))
        print(f"-> {sorted_occupancies}")
    
    print("--- End of Analysis ---")


def main():
    """
    Main function to run the script.
    Expects the CSV filename as a command-line argument.
    """
    # Check if a filename is provided
    if len(sys.argv) < 2:
        print("Usage: python analyze_occupancy.py <filename.csv>")
        print("\nExample: python analyze_occupancy.py sample_data.csv")
        # Automatically run with 'sample_data.csv' if it exists, for convenience
        if os.path.exists("sample_data.csv"):
            print("\nNo filename provided. Running with 'sample_data.csv' by default...")
            analyze_performance("sample_data.csv")
        return
    
    filename = sys.argv[1]
    analyze_performance(filename)

if __name__ == "__main__":
    main()