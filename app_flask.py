from flask import Flask, render_template, request
import joblib
import requests
import pandas as pd
from datetime import datetime
import os

# -- CONFIGURATION --
#API_KEY = "77b334c2dda8528f43217f7a408107bc"
API_KEY = os.environ.get("WEATHER_API_KEY")

app = Flask(__name__)

# --- 1. LOAD THE TRAINED MODEL ---
try:
    model= joblib.load('indian_weather_model.joblib')
except FileNotFoundError:
    print("Training File Not Found !!")
    exit()

def get_weather_type(conf_score, temp_min,temp_max, humidity, clouds, wind_speed):
    #hybrid app as it combines ML model + physics logic to returs emoji, status, description
    
    #1. Snow region
    if temp_min <=2 and conf_score >=50:
        return "❄️","Snow","Very cold conditions. Snowfall possible.."
    
    #2. Rain
    if conf_score >=62 and clouds>=40:
        desc ="Rain Likely, carry an Umbrella ☂️"
        
        if humidity >=80:
            desc += "Humid Monsoon conditions.."
        if wind_speed >=18:
            desc += "Gusty Winds Possible."
            
        return "🌧️","Rain",desc
    
    # Thunderstorms (common pre-monsoon)
    if conf_score >= 40 and humidity >= 70 and wind_speed >= 18:
        return "⛈️", "Thunderstorm", "Storms possible with lightning."

    # Fog (North India winters)
    if humidity >= 95 and wind_speed <= 7 and temp_min <= 18:
        return "🌫️", "Foggy", "Low visibility. Drive carefully."

    # Heatwave (common in Rajasthan, Delhi, central India)
    if temp_max >= 40:
        return "🔥", "Heatwave", "Extreme heat conditions."

    # Hot weather
    if temp_max >= 35:
        return "🌡️", "Hot", "Very warm day. Stay hydrated."

    # Cloud conditions
    if clouds >= 80:
        return "☁️", "Overcast", "Cloudy skies all day."

    if clouds >= 50:
        return "⛅", "Partly Cloudy", "Some clouds but mostly pleasant."

    # Clear sky
    return "☀️", "Sunny", "Clear skies."



@app.route('/')
def home():
    return render_template('index.html') #website

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        city = request.form['city']
        
        #for forecast 5 day rainfall 
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response=requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            forecast_list=[]
            
            #api sends 40 data points every 3 hour i.e; we want 1 point per day (like 12:00 pm)
            for item in data['list']:
                if "12:00:00" in item['dt_txt']:
                    #1.extract features -> model
                    features={
                        'MaxTemp':item['main']['temp_max'],
                        'MinTemp':item['main']['temp_min'],
                        'FeelsLike':item['main']['feels_like'],
                        'WindSpeed':item['wind']['speed']
                    }
                    #ask for model
                    df_input = pd.DataFrame([features])
                    prob = model.predict_proba(df_input)[0][1]
                    conf_score=int(prob*100)
                   
                    #get logic
                    emoji, status, desc = get_weather_type(
                        conf_score,
                        features["MinTemp"],
                        features["MaxTemp"],
                        item['main']['humidity'],
                        item['clouds']['all'],
                        features['WindSpeed']
                    )
                    #get / format date (i.e.; "2026-03-14" -> "Sat, 14 2026")
                    date_obj =datetime.strptime(item['dt_txt'], "%Y-%m-%d %H:%M:%S")
                    date_str = date_obj.strftime("%a, %d %b")
                    
                    #final: add into forecast_list
                    forecast_list.append({
                        'date':date_str,
                        'temp':round(item['main']['temp']),
                        'emoji':emoji,
                        'status':status,
                        'desc':desc,
                        'rain_chance':conf_score
                    })
                    # UX FIX:prevent visual contradictions
                    # If Physics forces "Sunny" or "Hot", but ML was wrongly predicting rain,
                    # we artificially lower the displayed risk so it makes sense to the user.
                    if status in ["Sunny", "Hot", "Heatwave"] and conf_score > 40:
                        conf_score = 15  # Hardcode a low, realistic rain chance
            
            return render_template('index.html', forecast=forecast_list, city=city) #Send the LIST of 5 days to HTML
        
        else:
            return render_template('index.html', error="City Not Found!", city=city)
                    
                    
if __name__ == '__main__':
    app.run(debug=True)