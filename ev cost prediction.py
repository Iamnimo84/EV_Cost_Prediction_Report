import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv("ev_charging_patterns.csv")
data_cleaned = data.dropna(subset=['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Cost (USD)', 'User Type'])
features = ['Energy Consumed (kWh)', 'Charging Duration (hours)']
target = 'Charging Cost (USD)'
X = data_cleaned[features]
y = data_cleaned[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression Results:")
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="b")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Charging Cost (USD)")
plt.ylabel("Predicted Charging Cost (USD)")
plt.grid(True)
plt.show()
