import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Smart Canteen AI",
    page_icon="🍽️",
    layout="wide"
)

# Load models
with open("demand_model.pkl", "rb") as file:
    demand_model = pickle.load(file)

with open("crowd_model.pkl", "rb") as file:
    crowd_model = pickle.load(file)

# Title
st.title("🍽️ AI-Powered Smart Canteen")
st.subheader("Demand and Crowd Prediction System")

st.write(
    "This system uses Machine Learning to predict canteen demand "
    "and crowd levels."
)

st.divider()

# Sidebar inputs
st.sidebar.header("Enter Canteen Details")

day = st.sidebar.selectbox(
    "Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

time_slot = st.sidebar.selectbox(
    "Time Slot",
    ["Breakfast", "Lunch", "Snacks", "Dinner"]
)

weather = st.sidebar.selectbox(
    "Weather",
    ["Sunny", "Cloudy", "Rainy"]
)

holiday = st.sidebar.selectbox(
    "Holiday",
    ["No", "Yes"]
)

special_event = st.sidebar.selectbox(
    "Special Event",
    ["No", "Yes"]
)

students = st.sidebar.number_input(
    "Expected Students",
    min_value=50,
    max_value=2000,
    value=500,
    step=50
)

holiday_value = 1 if holiday == "Yes" else 0
event_value = 1 if special_event == "Yes" else 0

input_data = pd.DataFrame({
    "Day": [day],
    "Time_Slot": [time_slot],
    "Weather": [weather],
    "Holiday": [holiday_value],
    "Special_Event": [event_value],
    "Students": [students]
})

# Prediction button
if st.button("🔮 Predict"):

    demand = demand_model.predict(input_data)[0]
    crowd = crowd_model.predict(input_data)[0]

    demand = max(0, round(demand))
    crowd = max(0, round(crowd))

    st.success("Prediction completed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🍱 Predicted Canteen Demand",
            f"{demand} meals"
        )

    with col2:
        st.metric(
            "👥 Predicted Crowd",
            f"{crowd} people"
        )

    # Crowd status
    if crowd < 300:
        status = "Low Crowd"
    elif crowd < 600:
        status = "Medium Crowd"
    else:
        status = "High Crowd"

    st.info(f"Canteen Status: **{status}**")

    # Chart
    st.subheader("Prediction Visualization")

    labels = ["Demand", "Crowd"]
    values = [demand, crowd]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Number of People / Meals")
    ax.set_title("Canteen Prediction")

    st.pyplot(fig)

st.divider()

st.subheader("💡 Benefits of the System")

st.write("""
• Helps the canteen prepare the required amount of food.

• Reduces food wastage.

• Helps manage crowded periods.

• Supports better staff planning.

• Helps improve the student dining experience.
""")