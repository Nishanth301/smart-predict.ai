import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data.csv")

# Inputs
X = data[["temperature", "vibration"]]

# Outputs
y = data["status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Create AI model
model = RandomForestClassifier()

# Train AI
model.fit(X_train, y_train)

# Test predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("AI Accuracy:", accuracy * 100, "%")

# User input
temp = float(input("Enter temperature: "))
vibration = float(input("Enter vibration level: "))

# Prediction
prediction = model.predict([[temp, vibration]])

print("Machine Status:", prediction[0])

import matplotlib.pyplot as plt

# Scatter plot
plt.scatter(data["temperature"], data["vibration"])

# Labels
plt.xlabel("Temperature")
plt.ylabel("Vibration")

# Title
plt.title("Machine Health Data")

# Show graph
plt.show()
