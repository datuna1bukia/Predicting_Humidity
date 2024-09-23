import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import serial
import time
import requests

API_KEY = 'd9cde27d8c4b887bc1ba00e53ada52d7'
LAT = 41.8251
LON = 41.8340

# Define the serial port (can be updated based on user input or config)
SERIAL_PORT = 'COM4'  # Default for Windows; update as needed

# Initialize the serial connection
arduino = serial.Serial(SERIAL_PORT, 9600)
time.sleep(2)

TARGET_HUMIDITY = 70  # Target soil humidity percentage

# Define the volume of the soil in cubic meters
SOIL_VOLUME_CUBIC_METERS = 0.006

# Load CSV files and prepare the model
weather_data = pd.read_csv('zugdidi_july_2024_weather.csv')
soil_data = pd.read_csv('zugdidi_july_2024_soil_humidity.csv')
data = pd.merge(soil_data, weather_data, on='Date')
data.fillna(method='ffill', inplace=True)
X = data[['Avg Temp (°C)', 'Max Temp (°C)', 'Min Temp (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Precipitation (mm)']]
y = data['Soil Humidity (%)']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MODEL TRAINING
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2)

def get_weather_data():
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp_kelvin = data['main']['temp']
        temp_celsius = temp_kelvin - 273.15
        temp_min_celsius = data['main']['temp_min'] - 273.15
        temp_max_celsius = data['main']['temp_max'] - 273.15
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        rain = data['rain'].get('1h', 0) if 'rain' in data else 0
        return pd.DataFrame({
            'Avg Temp (°C)': [temp_celsius],
            'Max Temp (°C)': [temp_max_celsius],
            'Min Temp (°C)': [temp_min_celsius],
            'Humidity (%)': [humidity],
            'Wind Speed (km/h)': [wind_speed * 3.6],
            'Precipitation (mm)': [rain]
        })
    else:
        raise Exception(f"Error fetching weather data (Status code: {response.status_code})")

def parse_arduino_data(line):
    try:
        # Split the line into components based on commas
        components = line.split(',')
        
        # STORE PARSED VALUE dictionary
        data = {}
        for component in components:
            key, value = component.split(':')
            data[key.strip()] = value.strip()
        
        # Extract soil humidity from the dictionary
        if 'SoilHumidity' in data:
            soil_humidity_raw = float(data['SoilHumidity'])
            return soil_humidity_raw
        else:
            raise ValueError("SoilHumidity key not found in the data")
    
    except ValueError as e:
        print(f"Error converting data: {e}")
        return None

def control_relay(soil_humidity, water_needed):
    if soil_humidity is not None:
        if soil_humidity < TARGET_HUMIDITY:
            arduino.write(b'1\n')  # Send '1' to turn on relay
            return f"Relay ON - Pump activated. Water needed: {water_needed:.2f} liters"
        else:
            arduino.write(b'0\n')  # Send '0' to turn off relay
            return "Relay OFF - Pump deactivated"
    else:
        return "No valid soil humidity data"

def calculate_water_needed(current_humidity, target_humidity, soil_volume):
    # Calculate the amount of water needed to reach the target humidity
    water_needed = (target_humidity - current_humidity) / 100 * soil_volume * 1000  # Convert cubic meters to liters
    return max(1, min(water_needed, 8))  # water_needed is between 1 and 8 liters

def main():
    try:
        while True:
            # Get current weather data
            new_weather_data = get_weather_data()

            # Standardize the new data using the same scaler
            new_weather_data_scaled = scaler.transform(new_weather_data)

            # Predict the soil humidity using current weather data
            predicted_humidity = model.predict(new_weather_data_scaled)[0][0]

            # Read and process Arduino data continuously
            if arduino.in_waiting > 0:
                while arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    print(f"Received from Arduino: {line}")  # Debug: print raw line received
                    
                    # Parse the soil humidity from the Arduino data
                    soil_humidity_raw = parse_arduino_data(line)

                    if soil_humidity_raw is not None:
                        # Calculate soil humidity percentage (adjust this calculation as needed)
                        soil_humidity = ((900 - soil_humidity_raw) / 500) * 100
                        
                        # Calculate water needed to reach the target humidity
                        water_needed = calculate_water_needed(soil_humidity, TARGET_HUMIDITY, SOIL_VOLUME_CUBIC_METERS)
                        
                        # Control the relay based on the soil humidity
                        relay_status = control_relay(soil_humidity, water_needed)

                        
                        print(f"Predicted Humidity: {predicted_humidity:.2f}%")
                        print(f"Soil Humidity (Raw): {soil_humidity:.2f}%")
                        print(relay_status)
                    else:
                        print("Failed to read soil humidity")

            #  the next iteration
            time.sleep(2)

    except KeyboardInterrupt:
        print("Program terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":  
    main()
