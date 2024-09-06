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

# Preprocessing (same as before)
data['price'] = data['price'].fillna(data['price'].median())
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data = data[(np.abs(data['price'] - data['price'].mean()) <= (3 * data['price'].std()))]

label_encoder = LabelEncoder()
data['Item Name'] = label_encoder.fit_transform(data['Item Name'])

# Model training
X = data[['year', 'month', 'day', 'Item Name']]
y = data['price']
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['GET'])
def predict_price():
    item_name = request.args.get('item')
    today = datetime.datetime.now()

    encoded_item_name = label_encoder.transform([item_name])[0]
    today_data = pd.DataFrame({
        'year': [today.year],
        'month': [today.month],
        'day': [today.day],
        'Item Name': [encoded_item_name]
    })
    
    today_data_scaled = scaler_X.transform(today_data)
    predicted_price_scaled = model.predict(today_data_scaled)
    predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
    
    return jsonify({
        'item': item_name,
        'predicted_price': predicted_price[0][0],
        'date': today.date().strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)
