import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\ASUS\Desktop\Gen AI\Sales Predection\train.csv")

data['Item_Fat_Content'] = data['Item_Fat_Content'].str.lower()  
data['Item_Fat_Content'] = data['Item_Fat_Content'].str.strip()  

data['Item_Fat_Content'].replace({
    'lf': 'low fat',
    'reg': 'regular',
    'low fat': 'low fat', 
}, inplace=True)

Outlet_Size = data['Outlet_Size'].value_counts()
print(Outlet_Size)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['Outlet_Size'] = label_encoder.fit_transform(data['Outlet_Size'])
data.head()

Outlet_Location_Type = data['Outlet_Location_Type'].value_counts()
print(Outlet_Location_Type)

label_encoder = LabelEncoder()
data['Outlet_Location_Type'] = label_encoder.fit_transform(data['Outlet_Location_Type'])
data.head()

Outlet_Type = data['Outlet_Type'].value_counts()
print(Outlet_Type)

label_encoder = LabelEncoder()
data['Outlet_Type'] = label_encoder.fit_transform(data['Outlet_Type'])
data.head()

Outlet_Identifier = data['Outlet_Identifier'].value_counts()
print(Outlet_Identifier)

label_encoder = LabelEncoder()
data['Outlet_Identifier'] = label_encoder.fit_transform(data['Outlet_Identifier'])
data.head()

data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year']

data.head()

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
data[data['Outlet_Establishment_Year'] == 1999]['Item_Outlet_Sales'].hist(alpha=0.5, label='1999', bins=10)
data[data['Outlet_Establishment_Year'] == 2000]['Item_Outlet_Sales'].hist(alpha=0.5, label='2000', bins=10)
plt.title('Sales by Outlet Establishment Year')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(2, 2, 2)
for size in data['Outlet_Size'].unique():
    data[data['Outlet_Size'] == size]['Item_Outlet_Sales'].hist(alpha=0.5, label=size, bins=10)
plt.title('Sales by Outlet Size')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(2, 2, 3)
for location in data['Outlet_Location_Type'].unique():
    data[data['Outlet_Location_Type'] == location]['Item_Outlet_Sales'].hist(alpha=0.5, label=location, bins=10)
plt.title('Sales by Outlet Location Type')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(2, 2, 4)
for outlet_type in data['Outlet_Type'].unique():
    data[data['Outlet_Type'] == outlet_type]['Item_Outlet_Sales'].hist(alpha=0.5, label=outlet_type, bins=10)
plt.title('Sales by Outlet Type')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

features = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
X = data[features]
y = data['Item_Outlet_Sales']

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X['Outlet_Size'] = label_encoder.fit_transform(X['Outlet_Size'])
X['Outlet_Location_Type'] = label_encoder.fit_transform(X['Outlet_Location_Type'])
X['Outlet_Type'] = label_encoder.fit_transform(X['Outlet_Type'])

X['Outlet_Age'] = 2024 - X['Outlet_Establishment_Year']
X.drop('Outlet_Establishment_Year', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_rf)
print(f"Mean Absolute Error: {mae}")

import joblib

joblib.dump(rf_model, 'random_forest_model.pkl')
