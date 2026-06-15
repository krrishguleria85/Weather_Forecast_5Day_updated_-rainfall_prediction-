import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# --- 1. LOAD DATASET ---
print("🔃 Loading the Dataset....")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'dataset', 'india_2000_2024_daily_weather.csv')

try:
  df = pd.read_csv(file_path)
  print(" File Found Successfully!! ")
except FileNotFoundError:
  print(" File not found! Make sure correct weather dataset exists..")
print()



# step 2: Check for some dataset info

print("Showing Dataset info: ")
df.info()
print()


print("\n🔍 Checking for Missing Values...")
print(df.isnull().sum())  # prints how many empty cells exist
print()


# --- STEP 3: PREPROCESSING
print("Steps 3 : Pre processing starts: ")

# 3.1. Rename columns to simple names
df = df.rename(columns={
    'temperature_2m_max': 'maxTemp',
    'temperature_2m_min': 'minTemp',
    'apparent_temperature_max': 'FeelsLikeMax',
    'apparent_temperature_min': 'FeelsLikeMin',
    'wind_speed_10m_max': 'WindSpeedMax',
    'wind_gusts_10m_max': 'WindGusts',
    'rain_sum': 'rainfall'
})
print()

print("Showing first rows data: \n", df.head(5))
print()

# keep date temp
df['date'] = pd.to_datetime(
    df['date'],
    dayfirst=True
)

#create seasonal date / month
df['Month'] = df['date'].dt.month
df['Dayofyear']= df['date'].dt.dayofyear

print(df[['date', 'Month', 'Dayofyear']].head())
print()


# encode the city
city_encoded = LabelEncoder()
df['cityEncoded'] = city_encoded.fit_transform(df['city'])
print(df['cityEncoded'].head(10))
print()



# Create the Target (RainTomorrow)
df['RainNextDay']= df['rainfall'].shift(-1)
df['RainTomorrow'] = ( df['RainNextDay'] > 0.1).astype(int)

# 3.2. DROP unnecessary columns 
df = df.drop(columns=['city', 'date', 'weather_code', 'precipitation_sum', 'wind_direction_10m_dominant'], errors='ignore')

df = df.dropna() #handle missing values


# then make a feature list for pred.

features = [
    'cityEncoded',
    
    'Month', 
    'Dayofyear',
    
    'maxTemp',
    'minTemp',

    'FeelsLikeMax',
    'FeelsLikeMin',

    'WindSpeedMax',
    'WindGusts',

]

# So training and testing the model first we find X and y

X = df[features]
y = df['RainTomorrow']

print(f"📊 Training on {len(X)} rows of data...")




#-- STEP 4: Training and Testing phase

# train model : time based split

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


#-- STEP 5: Train the Model by random forest classifier ------

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -- STEP 6: Evaluate the model ---

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "-" * 50)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 50)

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Detailed Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


print("\n🌟 Feature Importance")

for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")
    
    
    
#-- STEP 7: Saving the model --
joblib.dump(model,"indian_weather_model.joblib")
print(" Model saved as 'indian_weather_model.joblib'! ")
joblib.dump(city_encoded, "city_encoded.joblib")
