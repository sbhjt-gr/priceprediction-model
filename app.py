import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
from flask import Flask, request, jsonify
from fuzzywuzzy import process  # Import for fuzzy string matching

app = Flask(__name__)

# Initialize global variables
label_encoder = LabelEncoder()
scaler_X = StandardScaler()
model = LinearRegression()

# Function to process chunks of data and train the model
def train_model_in_chunks(chunk_size=10**5):
    global label_encoder, scaler_X, model
    # Process the dataset in chunks
    data_chunks = pd.read_csv('your_dataset.csv', chunksize=chunk_size)
    processed_chunks = [process_chunk(chunk) for chunk in data_chunks]
    data = pd.concat(processed_chunks)
    X = data[['year', 'month', 'day', 'Item Name']]
    y = data['price']
    X_scaled = scaler_X.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

# Function to process each data chunk
def process_chunk(chunk):
    chunk['price'] = chunk['price'].fillna(chunk['price'].median())
    chunk['Date'] = pd.to_datetime(chunk['Date'], dayfirst=True, errors='coerce')
    chunk = chunk.dropna(subset=['Date'])
    chunk['year'] = chunk['Date'].dt.year
    chunk['month'] = chunk['Date'].dt.month
    chunk['day'] = chunk['Date'].dt.day
    chunk['Item Name'] = chunk['Item Name'].str.lower()
    chunk = chunk[(np.abs(chunk['price'] - chunk['price'].mean()) <= (3 * chunk['price'].std()))]
    chunk['Item Name'] = label_encoder.fit_transform(chunk['Item Name'])
    return chunk

# Train the model during startup
train_model_in_chunks()

# Get unique item names for fuzzy matching
item_names_list = list(label_encoder.classes_)

@app.route('/predict', methods=['POST'])
def predict_price():
    content = request.json
    item_name = content.get('data', [None])[0]

    if not item_name:
        return jsonify({"error": "Item name not provided"}), 400

    item_name = item_name.lower()

    # Fuzzy matching to find the closest item name
    closest_match, confidence = process.extractOne(item_name, item_names_list)
    
    if confidence < 70:
        return jsonify({"error": f"No close match found for item '{item_name}'"}), 404

    encoded_item_name = label_encoder.transform([closest_match])[0]

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
        'searched_item': item_name,
        'matched_item': closest_match,
        'predicted_price': round(predicted_price[0], 2),
        'confidence': confidence,
        'date': today.date().strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
