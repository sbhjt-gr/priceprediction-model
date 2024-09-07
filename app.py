from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime

app = Flask(__name__)

# Load the preprocessed dataset (ensure this is stored within the project)
data = pd.read_csv('your_dataset.csv')

# Data Preprocessing
data['price'] = data['price'].fillna(data['price'].median())
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['Item Name'] = data['Item Name'].str.lower()
data = data[(np.abs(data['price'] - data['price'].mean()) <= (3 * data['price'].std()))]

# Encode 'Item Name' using LabelEncoder
label_encoder = LabelEncoder()
data['Item Name'] = label_encoder.fit_transform(data['Item Name'])

# Model training
X = data[['year', 'month', 'day', 'Item Name']]
y = data['price']
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict_price():
    content = request.json
    item_name = content.get('data', [None])[0]

    if not item_name:
        return jsonify({"error": "Item name not provided"}), 400

    item_name = item_name.lower()
    try:
        encoded_item_name = label_encoder.transform([item_name])[0]
    except ValueError:
        return jsonify({"error": f"Item '{item_name}' not found in the dataset"}), 404

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
