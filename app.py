import streamlit as st
import requests
import numpy as np
import pickle

# Load the model and necessary data
with open("logistic_regression_model.pkl", "rb") as file:
    data = pickle.load(file)

model = data['model']
scaler = data['scaler']
label_encoders = data['label_encoders']

# Streamlit UI for input
st.title("Booking Status Prediction")

# Get user inputs for each feature
no_of_adults = st.number_input("Number of Adults", min_value=0, value=1)
no_of_children = st.number_input("Number of Children", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, value=1)
no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, value=2)
type_of_meal_plan = st.selectbox("Type of Meal Plan", list(label_encoders['type_of_meal_plan'].classes_))
required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1])
room_type_reserved = st.selectbox("Room Type Reserved", list(label_encoders['room_type_reserved'].classes_))
lead_time = st.number_input("Lead Time", min_value=0, value=30)
arrival_year = st.selectbox("Arrival Year", [2017, 2018, 2019])
arrival_month = st.selectbox("Arrival Month", range(1, 13))
arrival_date = st.selectbox("Arrival Date", range(1, 32))
market_segment_type = st.selectbox("Market Segment Type", list(label_encoders['market_segment_type'].classes_))
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("No. of Previous Cancellations", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("No. of Previous Bookings Not Canceled", min_value=0, value=0)
avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, value=100.0)
no_of_special_requests = st.number_input("No. of Special Requests", min_value=0, value=0)

# Prepare the input data
input_data = [
    no_of_adults,
    no_of_children,
    no_of_weekend_nights,
    no_of_week_nights,
    label_encoders['type_of_meal_plan'].transform([type_of_meal_plan])[0],
    required_car_parking_space,
    label_encoders['room_type_reserved'].transform([room_type_reserved])[0],
    lead_time,
    arrival_year,
    arrival_month,
    arrival_date,
    label_encoders['market_segment_type'].transform([market_segment_type])[0],
    repeated_guest,
    no_of_previous_cancellations,
    no_of_previous_bookings_not_canceled,
    avg_price_per_room,
    no_of_special_requests
]

# Button to predict
if st.button("Predict Booking Status"):
    # Scale the input data
    scaled_input_data = scaler.transform([input_data])
    
    # Make prediction
    prediction = model.predict(scaled_input_data)

    if prediction[0] == 1:
        st.success("The booking is likely to be NOT Canceled.")
    else:
        st.error("The booking is likely to be Canceled.")

# Streamlit UI for CSV upload
st.title("Upload CSV to MongoDB")

# Upload CSV file widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Convert file to bytes for uploading
    file_bytes = uploaded_file.getvalue()
    
    # Prepare the request payload (use a multipart/form-data)
    files = {'file': (uploaded_file.name, uploaded_file, 'application/csv')}
    
    # Send a POST request to upload CSV file to Express server
    try:
        response = requests.post("http://localhost:3000/upload-csv", files=files)
        if response.status_code == 200:
            st.success("CSV file uploaded and data inserted into MongoDB successfully!")
        else:
            st.error("Failed to upload CSV file.")
    except Exception as e:
        st.error(f"Error: {e}")
