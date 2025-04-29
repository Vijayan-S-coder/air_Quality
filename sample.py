# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'C:\\Users\\CSE LAB\\Downloads\\AirQuality.csv'  # Change if needed
df = pd.read_csv(file_path, sep=';', decimal=',')  # The file uses ';' as separator and ',' as decimal

# Step 2: Basic cleaning
df = df.dropna(axis=1, how='all')  # Drop completely empty columns
df = df.dropna(axis=0, how='any')  # Drop rows with any NaN

# Replace -200 with NaN and then drop them
df.replace(-200, np.nan, inplace=True)
df.dropna(inplace=True)

# Step 3: Feature Selection
# Choose features and target
features = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]
target = 'C6H6(GT)'  # Benzene Concentration

X = df[features]
y = df[target]

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Step 8: Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Benzene (µg/m³)")
plt.ylabel("Predicted Benzene (µg/m³)")
plt.title("Actual vs Predicted Benzene Concentration")
plt.grid()
plt.show()
