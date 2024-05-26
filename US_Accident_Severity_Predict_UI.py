import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from scipy.sparse import hstack,csr_matrix
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
# Load the trained model
def load_predict_model(input_data):
    #st.write('Inside Load Model')
    model = joblib.load('Models/rf_model_classweights.joblib')
    with open('Models/feature_names.pkl', 'rb') as f:
        feature_names = joblib.load(f)
    # Preprocess the inputs
    data = preprocess_input(input_data,feature_names)
    # Predict the severity
    data.columns = data.columns.astype(str)
    prediction = model.predict(data)
    severity = prediction[0]
    st.success(f'The predicted severity of the accident is: Severity {severity}')

# Function to preprocess the inputs

def one_hot_encode_sparse(df, columns):
    #st.write('Inside one hot encoding')
    df = df.copy()
    sparse_matrices = []
    for column in columns:
        dummies = pd.get_dummies(df[column], sparse=True)
        sparse_matrices.append(csr_matrix(dummies))
        df = df.drop(column, axis=1)
    sparse_combined = hstack(sparse_matrices)
    df_sparse_encoded = pd.DataFrame.sparse.from_spmatrix(sparse_combined)
    df.reset_index(drop=True, inplace=True)
    df_sparse_encoded.reset_index(drop=True, inplace=True)
    df_combined = pd.concat([df, df_sparse_encoded], axis=1)
    return df_combined
def preprocess_input(data,feature_names):
    #st.write('Inside Preprocessing')
    cat_columns = ['Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight']
    categorical_columns_one_hot = ['Wind_Direction','Weather_Condition','Timezone']
    # Parse datetime
    data['Start_Time'] = pd.to_datetime(data['Start_Time'])
    data['Start_Hour'] = data['Start_Time'].dt.hour
    data['Start_Year'] = data['Start_Time'].dt.year
    data['Start_Day'] = data['Start_Time'].dt.day
    data['Start_Month'] = data['Start_Time'].dt.month
    data['Start_Minute'] = data['Start_Time'].dt.minute
    data = data.drop(columns=['Start_Time'])
    for column in cat_columns:
        data[column] = data[column].apply(lambda x: 1 if x == 'Day' else 0)
    df_accidents_sparse_encoded = one_hot_encode_sparse(data, categorical_columns_one_hot)
    binary_encoder = ce.binary.BinaryEncoder()
    city_binary_enc = binary_encoder.fit_transform(df_accidents_sparse_encoded["City"])
    df_accidents_sparse_encoded = pd.concat([df_accidents_sparse_encoded, city_binary_enc], axis=1).drop("City", axis=1)
    scaler = MinMaxScaler()
    numeric_cols = ['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Start_Lng','Start_Lat','Start_Year', 'Start_Month','Start_Day','Start_Hour','Start_Minute']
    df_accidents_sparse_encoded[numeric_cols] = scaler.fit_transform(df_accidents_sparse_encoded[numeric_cols])
    # Align columns to match training
    df_accidents_sparse_encoded = df_accidents_sparse_encoded.reindex(columns=feature_names, fill_value=0)
    df_accidents_sparse_encoded = df_accidents_sparse_encoded.drop(columns=['Severity'])
    return df_accidents_sparse_encoded

@st.cache_data
def load_cities():
    # User inputs
    #st.write('Inside Load Cities')
    # Load the data containing city names
    data = pd.read_csv('Data/US_Accidents_Dataset/US_Accidents_March23.csv')
    cities = data['City'].unique()
    return cities
cities = load_cities()
def main():
    # Streamlit UI
    st.title("US Accident Severity Prediction")

    st.markdown("""
    Enter the details of the accident to predict its severity.
    """)
    city = st.selectbox('City', cities)
    start_lng = st.number_input("Start Longitude")
    start_lat = st.number_input("Start Latitude")

    sunrise_sunset = st.selectbox("Sunrise/Sunset", ['Day', 'Night'])
    civil_twilight = st.selectbox("Civil Twilight", ['Day', 'Night'])
    nautical_twilight = st.selectbox("Nautical Twilight", ['Day', 'Night'])
    astronomical_twilight = st.selectbox("Astronomical Twilight", ['Day', 'Night'])

    start_date = st.date_input("Start Date")
    start_time = st.time_input("Start Time")
    timezone = st.selectbox("Timezone", ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific'])

    temperature = st.number_input("Temperature (F)")
    humidity = st.number_input("Humidity (%)")
    pressure = st.number_input("Pressure (in)")
    visibility = st.number_input("Visibility (mi)")
    wind_speed = st.number_input("Wind Speed (mph)")
    precipitation = st.number_input("Precipitation (in)")
    wind_direction = st.selectbox("Wind Direction", ['SW','CALM','W','N','S','NW','E','SE','VAR','NE'])
    weather_condition = st.selectbox("Weather Condition", ['Rain','Cloudy','Snow','Clear','Fog','Thunderstorm','Smoke','Windy',
    'Hail', 'Sand', 'Tornado'])

    # Add dropdowns or radio button groups for each feature
    crossing = st.radio('Crossing',("Yes", "No"), index=0)
    junction = st.radio('Junction',("Yes", "No"), index=0)
    railway = st.radio('Railway',("Yes", "No"), index=0)
    station = st.radio('Station',("Yes", "No"), index=0)
    stop = st.radio('Stop',("Yes", "No"), index=0)

    if st.button("Predict Severity"):
        # Create a DataFrame from user inputs
        #st.write('Inside button click')
        data = pd.DataFrame({
            'City': [city],
            'Wind_Direction': [wind_direction],
            'Weather_Condition': [weather_condition],
            'Timezone': [timezone],
            'Sunrise_Sunset': [sunrise_sunset],
            'Civil_Twilight': [civil_twilight],
            'Nautical_Twilight': [nautical_twilight],
            'Astronomical_Twilight': [astronomical_twilight],
            'Start_Time': [datetime.combine(start_date, start_time)],
            'Temperature(F)': [temperature],
            'Humidity(%)': [humidity],
            'Pressure(in)': [pressure],
            'Visibility(mi)': [visibility],
            'Wind_Speed(mph)': [wind_speed],
            'Precipitation(in)': [precipitation],
            'Start_Lng': [start_lng],
            'Start_Lat': [start_lat],
            'Crossing': [0 if crossing == "Yes" else 1],
            'Junction': [0 if junction == "Yes" else 1],
            'Railway': [0 if railway == "Yes" else 1],
            'Station': [0 if station == "Yes" else 1],
            'Stop': [0 if stop == "Yes" else 1]
        })
        #st.write(data)
        load_predict_model(data)
        
    

if __name__ == "__main__":
    main()