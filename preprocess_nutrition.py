import csv
import json
import re
import os

# Define I/O paths
CSV_PATH = 'assets/food_pair.csv'
JSON_PATH = 'assets/nutrition_db.json'

# This regex finds all component-amount-unit triples
NUTRITION_REGEX = re.compile(
    r"food component: (.*?) amount: (.*?) unit: (.*?)(?:\|\||$)"
)

def parse_nutrition_string(nutrition_str: str) -> dict:
    """
    Parses the complex nutrition string into a clean dictionary.
    """
    nutrients = {}
    matches = NUTRITION_REGEX.findall(nutrition_str)
    
    for match in matches:
        component, amount, unit = match
        
        # Clean up the component name to be a good JSON key
        # "Total lipid (fat)" -> "Total lipid (fat)_g"
        component_key = f"{component.strip()}_{unit.strip()}"
        
        try:
            # Convert amount to float
            amount_val = float(amount.strip())
        except ValueError:
            amount_val = 0.0 # Default to 0 if amount is not a number
            
        nutrients[component_key] = amount_val
        
    return nutrients

def preprocess_csv():
    """
    Reads the CSV, parses data, and saves it as a clean JSON file.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Error: Cannot find input file at {CSV_PATH}")
        return

    print(f"Starting preprocessing of {CSV_PATH}...")
    
    db_list = []
    
    try:
        with open(CSV_PATH, mode='r', encoding='utf-8') as infile:
            # Use csv.reader to handle quoted fields correctly
            reader = csv.reader(infile)
            
            # Skip header
            header = next(reader)
            print(f"Skipped header: {header}")
            
            count = 0
            for row in reader:
                if len(row) < 2:
                    continue # Skip empty or malformed rows
                    
                food_name = row[0].strip()
                nutrition_str = row[1]
                
                # Parse the complex string into a dict
                nutrition_data = parse_nutrition_string(nutrition_str)
                
                if not food_name or not nutrition_data:
                    print(f"Warning: Skipping row with empty name or data: {row}")
                    continue
                    
                db_list.append({
                    "food_name": food_name,
                    "nutrition": nutrition_data
                })
                count += 1

    except Exception as e:
        print(f"Error reading or parsing CSV: {e}")
        return

    # Ensure the assets directory exists
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    # Save the clean data to the new JSON file
    try:
        with open(JSON_PATH, 'w', encoding='utf-8') as outfile:
            json.dump(db_list, outfile, indent=2)
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return

    print(f"\nSuccess! Preprocessing complete.")
    print(f"Processed {count} food items.")
    print(f"Clean database saved to: {JSON_PATH}")

if __name__ == "__main__":
    preprocess_csv()