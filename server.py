from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests globally

@app.route('/get_location_names', methods=["GET"])
def get_location_names():
    """
    API endpoint to fetch all available locations.
    """
    try:
        locations = util.get_location_names()
        return jsonify({"locations": locations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_home_price', methods=["POST"])
def predict_home_price():
    """
    API endpoint to predict the price of a home based on input parameters.
    """
    try:
        # Retrieve input data
        location = request.form.get("location")
        total_sqft = request.form.get("total_sqft")
        bhk = request.form.get("bhk")
        bath = request.form.get("bath")

        # Validate input data
        if not location or not total_sqft or not bhk or not bath:
            return jsonify({"error": "Missing required parameters"}), 400

        # Convert inputs to proper types
        total_sqft = float(total_sqft)
        bhk = int(bhk)
        bath = int(bath)

        if total_sqft <= 0 or bhk <= 0 or bath <= 0:
            return jsonify({"error": "Input values must be positive"}), 400

        # Log inputs
        print(f"Inputs: location={location}, total_sqft={total_sqft}, bhk={bhk}, bath={bath}")

        # Predict the price
        predicted_price = util.predict_price(location, total_sqft, bath, bhk)
        print(f"Predicted price: {predicted_price}")

        # Return the prediction
        return jsonify({"predicted_price": predicted_price})
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure that the utility functions and artifacts are loaded before starting the server
    util.load_saved_artifacts()
    print("Starting Python Flask server on http://127.0.0.1:5000...")
    app.run(debug=True, port=5000)
