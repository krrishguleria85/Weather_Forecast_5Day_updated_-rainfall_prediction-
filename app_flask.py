from flask import Flask, render_template, request
import joblib
import requests
import pandas as pd
from datetime import datetime
import os



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
    
    # Thunderstorms (check before rain)
    if conf_score >= 60 and humidity >= 70 and wind_speed >= 18:
        return "⛈️", "Thunderstorm", "Storms possible with lightning."
    
    #2. Rain
    if conf_score >=70 and clouds>=50:
        desc ="Rain Likely, carry an Umbrella ☂️"
        
        if humidity >=80:
            desc += "Humid Monsoon conditions.."
        if wind_speed >=18:
            desc += "Gusty Winds Possible."
            
        return "🌧️","Rain",desc
    
    # Chance of rain
    if 50 <= conf_score < 70 and clouds >= 40:
        
        # Desert override
        if temp_max >= 33 and humidity < 60:
            return "☀️", "Sunny", "Hot desert conditions."
        
        return "🌦️", "Chance of Rain", "Rain possible later in the day."
    

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
        user_input = request.form['city']
        
        state_map = {
            "hp": "himachal", "up": "uttar pradesh", "mp": "madhya pradesh", 
            "ap": "andhra pradesh", "cg": "chhattisgarh", "mh": "maharashtra", 
            "rj": "rajasthan", "gj": "gujarat", "tn": "tamil nadu", "kl": "kerala", 
            "ka": "karnataka", "wb": "west bengal", "ts": "telangana", "pb": "punjab", 
            "hr": "haryana", "uk": "uttarakhand", "jh": "jharkhand", "br": "bihar", 
            "or": "odisha", "dl": "delhi", "jk": "jammu", "as": "assam",
        }
        
        # Separate the city from the state
        if "," in user_input:
            city_name = user_input.split(",")[0].strip()
            target_state = user_input.split(",")[1].strip().lower()
        else:
            words = user_input.strip().split()
            if len(words) > 1 and len(words[-1]) == 2:
                city_name = " ".join(words[:-1])
                target_state = words[-1].lower()
            else:
                # They just typed a normal city name
                city_name = user_input.strip()
                target_state = None
                
        if target_state in state_map:
            target_state = state_map[target_state]
            
        
        # STEP-> Highly Accurate Geocoding via api
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=20&language=en&format=json"
        geo_response = requests.get(geo_url).json()
        
        if "results" not in geo_response or len(geo_response["results"]) == 0:
            return render_template('index.html', error="City geographic data not found. Try another search.", city=user_input)
        
        
        # Look through the 5 results for the matching state
        selected_loc = geo_response["results"][0] # Default to the biggest one
        
        if target_state:
            for loc in geo_response["results"]:
                actual_state = loc.get("admin1", "").lower()
                if target_state in actual_state or actual_state in target_state:
                    selected_loc = loc
                    break
        
        
        # Extract exact map coordinates
        lat = selected_loc["latitude"]
        lon = selected_loc["longitude"]
        
        resolved_city_name = f"{selected_loc['name']}, {selected_loc.get('admin1', '')}"


        # Fetch Weather using Coordinates instead of Text ---
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response=requests.get(url)
        
        
        if response.status_code == 200:
            data = response.json()
            
            #to get weather at all time
            daily_data={}
            for item in data['list']:
                date_obj = datetime.strptime(item['dt_txt'], "%Y-%m-%d %H:%M:%S")
                date_key = date_obj.strftime("%Y-%m-%d")
                date_display = date_obj.strftime("%a, %d %b")
                
                if date_key not in daily_data:
                    daily_data[date_key]={
                        'date':date_display,
                        'temps':[],
                        'min_temps':[],
                        'max_temps':[],
                        'feels_like': [],
                        'wind_speeds': [],
                        'humidities': [],
                        'clouds': []
                    }
                # Collect values from all times of the day
                daily_data[date_key]['temps'].append(item['main']['temp'])
                daily_data[date_key]['min_temps'].append(item['main']['temp_min'])
                daily_data[date_key]['max_temps'].append(item['main']['temp_max'])
                daily_data[date_key]['feels_like'].append(item['main']['feels_like'])
                daily_data[date_key]['wind_speeds'].append(item['wind']['speed'])
                daily_data[date_key]['humidities'].append(item['main']['humidity'])
                daily_data[date_key]['clouds'].append(item['clouds']['all'])
            
            
            
            forecast_list=[]
            for date_key, metrics in list(daily_data.items())[:5]: #limit to 5 days
                
                #cal. true day summaries across all time zomes
                true_max = max(metrics['max_temps'])
                true_min = min(metrics['min_temps'])
                avg_feels = sum(metrics['feels_like']) / len(metrics['feels_like'])
                avg_wind = sum(metrics['wind_speeds']) / len(metrics['wind_speeds'])
                avg_humidity = sum(metrics['humidities']) / len(metrics['humidities'])
                avg_clouds = sum(metrics['clouds']) / len(metrics['clouds'])
                
                #1. create a features dataframe for model
                features={
                    'MaxTemp':true_max,
                    'MinTemp':true_min,
                    'FeelsLike':avg_feels,
                    'WindSpeed':avg_wind
                }
                df_input = pd.DataFrame([features])
                prob = model.predict_proba(df_input)[0][1]
                conf_score = int(prob * 100)
                
                #2. run rule based logic
                emoji, status, desc = get_weather_type(
                    conf_score,
                    true_min,
                    true_max,
                    avg_humidity,
                    avg_clouds,
                    avg_wind
                )
                
                #3. apply the clean ui rules
                if status in ["Sunny", "Hot", "Heatwave"] and conf_score > 40:
                    conf_score = 15
                if conf_score<20:
                    conf_score=0
                    
                #append true day peaks into final forcast values
                forecast_list.append({
                    'date': metrics['date'],
                    'temp': round(sum(metrics['temps']) / len(metrics['temps'])),
                    'high_temp': round(true_max),
                    'low_temp': round(true_min),
                    'emoji':emoji,
                    'status':status,
                    'desc':desc,
                    'rain_chance':conf_score
                })
                
            return render_template('index.html', forecast=forecast_list, city=resolved_city_name) 
        
        else:
            return render_template('index.html', error="City Not Found!", city=user_input)
    
                    
if __name__ == '__main__':
    app.run(debug=True)