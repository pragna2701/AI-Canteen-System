import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

days = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

time_slots = ["Breakfast", "Lunch", "Snacks", "Dinner"]
weather = ["Sunny", "Cloudy", "Rainy"]

data = {
    "Day": np.random.choice(days, n),
    "Time_Slot": np.random.choice(time_slots, n),
    "Weather": np.random.choice(weather, n),
    "Holiday": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    "Special_Event": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    "Students": np.random.randint(200, 1000, n)
}

df = pd.DataFrame(data)

demand = []

for _, row in df.iterrows():

    value = row["Students"] * 0.45

    if row["Time_Slot"] == "Lunch":
        value += 100
    elif row["Time_Slot"] == "Breakfast":
        value += 30
    elif row["Time_Slot"] == "Snacks":
        value += 60
    else:
        value += 50

    if row["Holiday"] == 1:
        value -= 100

    if row["Special_Event"] == 1:
        value += 150

    if row["Weather"] == "Rainy":
        value -= 30

    value += np.random.normal(0, 30)

    demand.append(max(20, int(value)))

df["Demand"] = demand

df["Crowd"] = (
    df["Students"] * 0.55
    + df["Special_Event"] * 120
    - df["Holiday"] * 100
    + np.random.normal(0, 25, n)
).astype(int)

df["Crowd"] = df["Crowd"].clip(lower=10)

df.to_csv("canteen_data.csv", index=False)

print("Canteen dataset created successfully!")
print(df.head())