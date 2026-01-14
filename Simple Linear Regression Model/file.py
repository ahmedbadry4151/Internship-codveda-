# Simple Linear Regression Model for Stock Prices Data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
print("Loading data...")
data = pd.read_csv("Stock Prices Data Set.csv")
print(f"Dataset Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Drop missing values
data = data.dropna()
print(f"Shape after cleaning: {data.shape}")

# Feature Engineering
data['price_range'] = data['high'] - data['low']
data['price_change'] = data['close'] - data['open']
data['avg_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

# Encode symbol
le = LabelEncoder()
data['symbol_encoded'] = le.fit_transform(data['symbol'])

# Prepare features and target
feature_cols = ['open', 'high', 'low', 'volume', 'price_range', 'avg_price', 'symbol_encoded']
target_col = 'close'

X = data[feature_cols]
y = data[target_col]

print(f"Features: {feature_cols}")
print(f"Target: {target_col}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
print("\nTraining Linear Regression Model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluation
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Training MSE: {train_mse:.4f}")
print(f"Training R2:  {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"Test MSE:     {test_mse:.4f}")
print(f"Test R2:      {test_r2:.4f} ({test_r2*100:.2f}%)")

# Feature Coefficients
print("\n========== COEFFICIENTS ==========")
for feat, coef in zip(feature_cols, model.coef_):
    print(f"{feat:20s}: {coef:12.6f}")
print(f"{'Intercept':20s}: {model.intercept_:12.6f}")

# Model Quality
print("\n========== INTERPRETATION ==========")
if test_r2 > 0.9:
    print("Excellent model! R2 > 90%")
elif test_r2 > 0.7:
    print("Good model! R2 > 70%")
else:
    print("Model needs improvement.")