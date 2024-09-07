import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
from flask import Flask, request, jsonify
from difflib import get_close_matches  # For finding closest matches

app = Flask(__name__)

# Initialize global variables
label_encoder = LabelEncoder()
scaler_X = StandardScaler()
model = LinearRegression()

# Function to process chunks of data
def process_chunk(chunk):
    # Data Preprocessing
    chunk['price'] = chunk['price'].fillna(chunk['price'].median())
    chunk['Date'] = pd.to_datetime(chunk['Date'], dayfirst=True, errors='coerce')
    chunk = chunk.dropna(subset=['Date'])
    chunk['year'] = chunk['Date'].dt.year
    chunk['month'] = chunk['Date'].dt.month
    chunk['day'] = chunk['Date'].dt.day
    chunk['Item Name'] = chunk['Item Name'].str.lower()
    chunk = chunk[(np.abs(chunk['price'] - chunk['price'].mean()) <= (3 * chunk['price'].std()))]
    
    # Encode 'Item Name'
    chunk['Item Name'] = label_encoder.fit_transform(chunk['Item Name'])
    
    return chunk

# Load and preprocess data in chunks
data_chunks = pd.read_csv('your_dataset.csv', chunksize=10**6)
data = pd.concat([process_chunk(chunk) for chunk in data_chunks])

# Store the original item names for matching later
original_item_names = data['Item Name']

# Model training
X = data[['year', 'month', 'day', 'Item Name']]
y = data['price']
X_scaled = scaler_X.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict_price():
    content = request.json
    item_name = content.get('data', [None])[0]

    if not item_name:
        return jsonify({"error": "Item name not provided"}), 400

    item_name = item_name.lower()

    # Check for an exact match first
    try:
        encoded_item_name = label_encoder.transform([item_name])[0]
    except ValueError:
        # Try to find the closest matching item
        closest_matches = get_close_matches(item_name, original_item_names, n=1, cutoff=0.6)
        if not closest_matches:
            return jsonify({"error": f"Item '{item_name}' not found in the dataset"}), 404
        closest_item_name = closest_matches[0]
        encoded_item_name = label_encoder.transform([closest_item_name])[0]
        item_name = closest_item_name  # Update to the matched item name

    # Get today's date
    today = datetime.datetime.now()
    today_data = pd.DataFrame({
        'year': [today.year],
        'month': [today.month],
        'day': [today.day],
        'Item Name': [encoded_item_name]
    })

    today_data_scaled = scaler_X.transform(today_data)
    predicted_price = model.predict(today_data_scaled)

    return jsonify({
        'item': item_name,
        'predicted_price': round(predicted_price[0], 2),
        'date': today.date().strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
