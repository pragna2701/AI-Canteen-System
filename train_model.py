import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("canteen_data.csv")

X = df[
    [
        "Day",
        "Time_Slot",
        "Weather",
        "Holiday",
        "Special_Event",
        "Students"
    ]
]

y_demand = df["Demand"]
y_crowd = df["Crowd"]

categorical = ["Day", "Time_Slot", "Weather"]
numerical = ["Holiday", "Special_Event", "Students"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="passthrough"
)

# Demand model
demand_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_demand, test_size=0.2, random_state=42
)

demand_model.fit(X_train, y_train)

demand_prediction = demand_model.predict(X_test)

print(
    "Demand MAE:",
    round(mean_absolute_error(y_test, demand_prediction), 2)
)

# Crowd model
crowd_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_crowd, test_size=0.2, random_state=42
)

crowd_model.fit(X_train, y_train)

crowd_prediction = crowd_model.predict(X_test)

print(
    "Crowd MAE:",
    round(mean_absolute_error(y_test, crowd_prediction), 2)
)

# Save models
with open("demand_model.pkl", "wb") as file:
    pickle.dump(demand_model, file)

with open("crowd_model.pkl", "wb") as file:
    pickle.dump(crowd_model, file)

print("Models trained and saved successfully!")