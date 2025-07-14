import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Loading the data
df = pd.read_csv('D:\Internship projects\car data.csv')

# Step 2: Viewing basic info
print("First 5 Rows:\n", df.head())
print("\nDataset Info:")
print(df.info())

# Step 3: Preprocessing
# Renaming columns 
df.columns = [col.strip() for col in df.columns]

# Checking for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Converting categorical variables using LabelEncoder
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Step 4: Feature Selection
# Assuming 'Selling_Price' is the target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Step 5: Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluating
print("\nModel Performance:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 9: Feature Importance
plt.figure(figsize=(10, 5))
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Step 10: Saving the model
joblib.dump(model, 'car_price_model.pkl')
print("\nModel saved as 'car_price_model.pkl'")
