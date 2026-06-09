import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. LOAD DATASET ---
print("🔃 Loading the Dataset....")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'dataset', 'india_2000_2024_daily_weather.csv')

try:
  df = pd.read_csv(file_path)
  print("✅ File Found Successfully!! ")
except FileNotFoundError:
  print("❌ File not found! Make sure correct weather dataset exists..")
  


# --- STEP 2: CHECK FOR MISSING VALUES ---
print("\n🔍 Checking for Missing Values...")
print(df.isnull().sum())  # prints how many empty cells exist



# --- STEP 3: PREPROCESSING (Cleaning) ---
print("\n🧹 Cleaning and Preparing Data...")

# 3.1. Rename columns to simple names
df = df.rename(columns={
    'temperature_2m_max': 'MaxTemp',
    'temperature_2m_min': 'MinTemp',
    'apparent_temperature_max': 'FeelsLike',
    'wind_speed_10m_max': 'WindSpeed',
    'rain_sum': 'Rainfall'
})

# 3.2. DROP unnecessary columns like, We remove 'city', 'date', 'weather_code', etc. because we want a Universal Model.
df = df.drop(columns=['city', 'date', 'weather_code', 'precipitation_sum','apparent_temperature_min', 'wind_gusts_10m_max', 'wind_direction_10m_dominant'], errors='ignore') 


# 3.3. Create the Target (RainTomorrow)
# Shift rainfall up by 1 row to predict "Next Day"
df['RainNextDay'] = df['Rainfall'].shift(-1)
df['RainTomorrow'] = (df['RainNextDay'] > 0.1).astype(int)


# 4. HANDLE MISSING VALUES:- Since we have 90,000+ rows, dropping a few empty ones is safe and accurate.
df = df.dropna()

print("✅ Data Cleaned & Ready!")
print(f"Final Shape: {df.shape}")  # Shows (Rows, Columns)
print(df.head())
print("\n")


#-- so use these 4 features to predict
features =['MaxTemp','MinTemp','FeelsLike','WindSpeed']
X = df[features]
y = df['RainTomorrow']


# Remove empty rows (created by shifting or missing data); so, This aligns X and y perfectly
data_clean = pd.concat([X, y], axis=1).dropna()
X = data_clean[features]
y = data_clean['RainTomorrow']

print(f"📊 Training on {len(X)} rows of data...")




#-- STEP 4: Training and Testing phase (Train - Test Split) ----------
#so in this, splitting the model training into two phases: 80% in training set and rest 20% in testing set

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)


#-- STEP 5: Train the Model by random forest classifier ------
print("🚀 Training the Random Forest Model Starts ")
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state= 42, n_jobs= -1)
model.fit(X_train, y_train)
print("\n")


# -- STEP 6: Evaluate the model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("------------------------------------------------")
print(f" Model Accuracy: {accuracy * 100:.2f}%")
print("------------------------------------------------")
print("Detailed Report:\n", classification_report(y_test, y_pred))
print("\n")


#-- STEP 7: Saving the model --
joblib.dump(model,"indian_weather_model.joblib")
print(" Model saved as 'indian_weather_model.joblib'! ")
