# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime

# Load dataset
data = pd.read_csv('your_dataset.csv')

# Data Preprocessing
# Handle missing values (e.g., fill missing prices with the median price)
data['price'] = data['price'].fillna(data['price'].median())

# Convert 'Date' to datetime format with dayfirst=True for day/month/year format
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Drop rows where the date conversion failed
data = data.dropna(subset=['Date'])

# Extract useful features from the date
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

# Detect and remove outliers in the price (optional, depending on dataset)
# Remove prices that are beyond 3 standard deviations from the mean
data = data[(np.abs(data['price'] - data['price'].mean()) <= (3 * data['price'].std()))]

# Encode categorical variables (Item Name)
label_encoder = LabelEncoder()
data['Item Name'] = label_encoder.fit_transform(data['Item Name'])

# Define features (X) and target (y)
X = data[['year', 'month', 'day', 'Item Name']]
y = data['price']

# Feature scaling (StandardScaler) to scale both X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize the model (Linear Regression)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_scaled = model.predict(X_test)

# Inverse transform the predictions to get back to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate the model (using Mean Squared Error)
mse = mean_squared_error(scaler_y.inverse_transform(y_test), y_pred)
print(f'Mean Squared Error: {mse}')

# Predict price for each vegetable as of today's date
today = datetime.datetime.now()

# Decode the item names for prediction
item_names = label_encoder.inverse_transform(data['Item Name'].unique())

# Predict the price for each vegetable for today's date
predictions = {}
for item_name in item_names:
    encoded_item_name = label_encoder.transform([item_name])[0]
    today_data = pd.DataFrame({
        'year': [today.year],
        'month': [today.month],
        'day': [today.day],
        'Item Name': [encoded_item_name]
    })
    
    # Scale the input data for prediction
    today_data_scaled = scaler_X.transform(today_data)
    
    # Predict and inverse transform the price
    predicted_price_scaled = model.predict(today_data_scaled)
    predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
    
    predictions[item_name] = predicted_price[0][0]  # Price per kilogram

# Display predicted prices for all vegetables (per kilogram)
for vegetable, price in predictions.items():
    print(f'Predicted Price of {vegetable} on {today.date()} per kilogram: {price}')
