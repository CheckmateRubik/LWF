import requests
import datetime,pytz
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
from datetime import timedelta
from flask import Flask, jsonify


app = Flask(__name__)


# fetch weather API

url = "https://playground.kid-bright.org/api/data_table_wide"
stations = "256510748567,240AC4AADE1C"
format_type = "series"
headers = {"User-Agent": "GOD TEERAWAT"}

# def the data fuckerrrr

def weatherforecaster(zawarudo, apichit):
  current_time = datetime.datetime.now(pytz.timezone('Asia/Bangkok'))
  api_starttime = (current_time - timedelta(hours=zawarudo)).strftime('%Y-%m-%d %H:%M:%S')
  api_endtime = current_time.strftime('%Y-%m-%d %H:%M:%S')
  querystring = {"": "", "stations": stations, "format": format_type, "start": api_starttime, "end": api_endtime, "freq": apichit}
  response = requests.get(url, headers=headers, params=querystring)

  json_data = response.json()
  columns = json_data["columns"]
  dat = json_data["data"]
  df = pd.DataFrame(dat, columns=columns)
  df['temp'] = df['temp'].fillna(df['temp'].mean())
  df['humid'] = df['humid'].fillna(df['humid'].mean())
  df['rainfall'] = df['rainfall'].fillna(df['rainfall'].mean())
  df['wind_direct'] = df['wind_direct'].fillna(df['wind_direct'].mean())
  df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
  data = df

  # temp

  model_temp = load_model('lstm_temp.keras')
  scaler_temp = joblib.load('scaler_temp.pkl')

  prediction_data = data
  features = ['temp']
  sequence_length = 6

  predicted_features = model_temp.predict(prediction_data[features].tail(sequence_length).values.reshape(1, sequence_length, -1))
  predicted_features = scaler_temp.inverse_transform(predicted_features)
  temp_condition =  (predicted_features)[0][0]

  # humid

  model_humid = load_model('lstm_Humid.keras')
  scaler_humid = joblib.load('scaler_humid.pkl')

  prediction_data = data
  features = ['humid']
  sequence_length = 6

  predicted_features = model_humid.predict(prediction_data[features].tail(sequence_length).values.reshape(1, sequence_length, -1))
  predicted_features = scaler_humid.inverse_transform(predicted_features)
  humid_condition =  (predicted_features)[0][0]

  #rainfall

  model_rf = load_model('lstm_prec.keras')
  scaler_rf = joblib.load('scaler_rainfall.pkl')

  prediction_data = data
  features = ['rainfall']
  sequence_length = 6

  predicted_features = model_rf.predict(prediction_data[features].tail(sequence_length).values.reshape(1, sequence_length, -1))
  predicted_features = scaler_rf.inverse_transform(predicted_features)
  rainfall_condition =  (predicted_features)[0][0]

  #wind_direct

  model_wd = load_model('lstm_WD.keras')
  scaler_wd = joblib.load('scaler_wd.pkl')

  prediction_data = data
  features = ['wind_direct']
  sequence_length = 6

  predicted_features = model_wd.predict(prediction_data[features].tail(sequence_length).values.reshape(1, sequence_length, -1))
  predicted_features = scaler_wd.inverse_transform(predicted_features)
  wind_direct_condition =  (predicted_features)[0][0]

  #wind_speed

  model_ws = load_model('lstm_WS.keras')
  scaler_ws = joblib.load('scaler_ws.pkl')

  prediction_data = data
  features = ['wind_speed']
  sequence_length = 6

  predicted_features = model_ws.predict(prediction_data[features].tail(sequence_length).values.reshape(1, sequence_length, -1))
  predicted_features = scaler_ws.inverse_transform(predicted_features)
  wind_speed_condition =  (predicted_features)[0][0]

  # end of training data

  temp_condition = np.round(temp_condition, 1)
  humid_condition = np.round(humid_condition, 1)
  rainfall_condition = np.round(rainfall_condition, 1)
  wind_speed_condition = np.round(wind_speed_condition, 1)


  if 0 <= wind_speed_condition < 22.5 or wind_speed_condition >= 337.5:
      winddirection = "north (N)"
  elif 22.5 <= wind_speed_condition < 45:
      winddirection = "north-northeast (NNE)"
  elif 45 <= wind_speed_condition < 67.5:
      winddirection = "northeast (NE)"
  elif 67.5 <= wind_speed_condition < 90:
      winddirection = "east-northeast (ENE)"
  elif 90 <= wind_speed_condition < 112.5:
      winddirection = "east (E)"
  elif 112.5 <= wind_speed_condition < 135:
      winddirection = "east-southeast (ESE)"
  elif 135 <= wind_speed_condition < 157.5:
      winddirection = "southeast (SE)"
  elif 157.5 <= wind_speed_condition < 180:
      winddirection = "south-southeast (SSE)"
  elif 180 <= wind_speed_condition < 202.5:
      winddirection = "south (S)"
  elif 202.5 <= wind_speed_condition < 225:
      winddirection = "south-southwest (SSW)"
  elif 225 <= wind_speed_condition < 247.5:
      winddirection = "southwest (SW)"
  elif 247.5 <= wind_speed_condition < 270:
      winddirection = "west-southwest (WSW)"
  elif 270 <= wind_speed_condition < 292.5:
      winddirection = "west (W)"
  elif 292.5 <= wind_speed_condition < 315:
      winddirection = "west-northwest (WNW)"
  else:
      winddirection = "northwest (NW)"

  # rain predict by KEE YAA BEST


  return wind_speed_condition, rainfall_condition, temp_condition



@app.route('/', methods=['GET'])
def get_weather():
    oneday = [float(value) for value in weatherforecaster(720, "1440mins")]
    fifteenmin = [float(value) for value in weatherforecaster(6, "10mins")]
    thirtymin = [float(value) for value in weatherforecaster(36, "30mins")]
    onehour = [float(value) for value in weatherforecaster(256, "60mins")]
    threehour = [float(value) for value in weatherforecaster(180, "180mins")]
    sixhour = [float(value) for value in weatherforecaster(360, "360mins")]
    twelvehour = [float(value) for value in weatherforecaster(720, "720mins")]

    return jsonify({
        "oneday": oneday,
        "fifteenmin": fifteenmin,
        "thirtymin": thirtymin,
        "onehour": onehour,
        "threehour": threehour,
        "sixhour": sixhour,
        "twelvehour": twelvehour
        })

if __name__ == "__main__":
    app.run()
