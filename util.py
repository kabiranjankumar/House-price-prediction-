import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

import warnings

def predict_price(location, total_sqft, bath, bhk):
    """
    Predicts the price of a house based on location, sqft, bath, and bhk.
    """
    if not __data_columns:
        return "Model artifacts not loaded. Please load artifacts first."

    try:
        location_index = __data_columns.index(location.lower())
    except ValueError:
        return f"Location '{location}' not found in the dataset."

    # Create feature vector
    features = np.zeros(len(__data_columns))
    features[0] = total_sqft
    features[1] = bath
    features[2] = bhk
    if location_index >= 0:
        features[location_index] = 1

    # Suppress UserWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        predicted_price = __model.predict([features])[0]

    return round(predicted_price, 2)

def get_location_names():
    """
    Returns the list of available locations.
    """
    if __locations:
        return __locations
    else:
        return []

def load_saved_artifacts():
    """
    Loads model and column artifacts from disk.
    """
    print("Loading saved artifacts...")
    global __data_columns
    global __locations
    global __model

    try:
        # Load data columns
        with open(r"C:\bhk\server\artifacts\columns.json", "r") as f:
            data = json.load(f)
            __data_columns = data.get("data_columns", [])
            if not __data_columns:
                raise ValueError("No data columns found in columns.json.")
            
            __locations = [x.title() for x in __data_columns[3:]]  # Title-case for consistent display
            print(f"Loaded data columns: {__data_columns}")
            print(f"Extracted locations: {__locations}")

        # Load model
        with open(r"C:\bhk\server\artifacts\banglore_home_price_model.pickle", "rb") as f:
            __model = pickle.load(f)
            print("Model loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except ValueError as e:
        print(f"Error: Invalid file content - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    load_saved_artifacts()
    print("Available locations:", get_location_names())
    print("Predicted price:", predict_price("Vijayanagar", 1000, 3, 3))
