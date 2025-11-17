# ============================================================
# Importing Required Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 🔧 1. LINEAR REGRESSION (SINGLE VARIABLE)
# ============================================================

# Define data
data = {
    'Area': [2600,3000,3200,3600,4000],
    'price': [550000,565000,610000,680000,725000]
}
df = pd.DataFrame(data)
print(df)

# ------------------------------------------------------------
# ✨ Set local path for saving CSV
# ------------------------------------------------------------
house_price_path = "/home/adarsh/Desktop/ml_l_class_ref/files/house_price.csv"   # <— CHANGE THIS

# index false means do not create extra col for serial no  , savin the csv to disk , write cmd 
df.to_csv(house_price_path, index=False)



# Load the same CSV
df = pd.read_csv(house_price_path)
print(df.head())




# ------------------------------------------------------------
# Scatter Plot – Area vs Price
# ------------------------------------------------------------
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area, df.price, color='red', marker='+')
plt.show()

# ------------------------------------------------------------
# Train Linear Regression Model
# ------------------------------------------------------------
reg = linear_model.LinearRegression()
reg.fit(df[['Area']], df.price)

# Prediction for given area
print("Predicted price for area 3300:", reg.predict([[3300]])[0])

# Print model parameters
print("Coefficient:", reg.coef_[0])
print("Intercept:", reg.intercept_)

# Manual check (optional)
print("Manual Calculation:", 135.78767123 * 3300 + 180616.43835616432)

# ------------------------------------------------------------
# Plot Regression Line
# ------------------------------------------------------------
plt.scatter(df.Area, df.price, color='red', marker='+')
plt.plot(df.Area, reg.predict(df[['Area']]), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title("Linear Regression - Single Variable")
plt.show()


# ============================================================
# 🔧 2. LINEAR REGRESSION (MULTIPLE VARIABLE)
# Bottle Dataset
# ============================================================

# ------------------------------------------------------------
# 🔥 Set local bottle dataset path
# ------------------------------------------------------------
bottle_path = "/home/adarsh/Desktop/ml_l_class_ref/files/bottle.csv"    # <— UNZIP AND POINT TO THE CSV

df = pd.read_csv(bottle_path)

print(df.columns)
print(df.shape)

# Select salinity & temperature
df_temp = df[['Salnty','T_degC']]
df_temp.columns = ['salinity', 'temperature']
print(df_temp.head(30))

# Scatter plot
plt.scatter(df_temp.salinity, df_temp.temperature, color='red', marker='+')
plt.xlabel("Salinity")
plt.ylabel("Temperature")
plt.title("Bottle Dataset")
plt.show()

# ------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------
df_temp.ffill(inplace=True)

# Prepare training data
X = df_temp['salinity'].values.reshape(-1,1)
y = df_temp['temperature'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Train model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print('\n==Model Evaluation==')
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R² Score:', r2)

# Regression plot
plt.scatter(df_temp.salinity, df_temp.temperature, color='red', marker='+')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.title('Bottle Dataset Regression')
plt.show()

# ============================================================
# 🔧 3. REGRESSION ON FIRST 500 ROWS
# ============================================================

df_new = df_temp[:500]
X = df_new['salinity'].values.reshape(-1,1)
y = df_new['temperature'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.scatter(df_new.salinity, df_new.temperature, color='red', marker='+')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.title("Bottle Dataset Regression (First 500 Rows)")
plt.show()

# Recalculate evaluation metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print('\n==Model Evaluation (First 500 rows)==')
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R² Score:', r2)


# ============================================================
# Tasks for you:
#   • Join two DataFrames
#   • Use groupby()
#   • Cleaning textual datasets
# ============================================================
