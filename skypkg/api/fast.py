# Imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Load Pipelines from pickle files
delay_pipeline = pickle.load(open("skypkg/model_pickle_files/flight_delay_pipeline_log_smote_final.pkl", "rb"))
demand_forecast_pipeline = pickle.load(open("skypkg/model_pickle_files/pipeline_knn2_tuned.pkl", "rb"))


# Load lookup tables

flight_details_table = pd.read_csv("data/flight-detail-grouped.csv")
city_details_table = "ipsum lorem"



#Initialize FastAPI

app = FastAPI()



# Delay probability endpoint
@app.get("/delay_proba")
def predict_delay(
        flight_number: int,  # 1960
        departure_month: int # 1
):
    """
    Predicts the probability of flight delay for a given flight_number and departure_month.

    Parameters:
        flight_number (int): The flight number to search for.
        departure_month (int): The departure month (e.g., 1 for January).
    OBS:
        flight_details_table initialized internally in fast.py

    Returns:
        float: The predicted probability of a delay.

    """

    #### get flight detail

    # Search for the relevant flight details
    try:
        flight_detail = flight_details_table.loc[flight_details_table['flight_number'] == flight_number].copy()

        if flight_detail.empty:
            return f"No details found for flight number {flight_number}"

        # Add input departure_month as a new column
        flight_detail['departure_month'] = departure_month

        # This is the data that goes into pipeline for prediction
        X_new001 = flight_detail.copy()
        # Predict the probability of delay
        predicted_proba = delay_pipeline.predict_proba(X_new001)[:, 1]
        binary_pred = delay_pipeline.predict(X_new001)
        if binary_pred == 1:
            outcome = "Delayed"
        if binary_pred == 0:
            outcome = "On time"


        return: dict()


    f"Probability of Delay : {predicted_proba}. Flight is likely to be: {outcome}"










@app.get("/demand_forecast")






@app.get("/pricing_forecast")

### Input 

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END






############################


    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    X_pred = pd.DataFrame(locals(), index=[0])

    # Convert to US/Eastern TZ-aware!
    X_pred['pickup_datetime'] = pd.Timestamp(pickup_datetime, tz='US/Eastern')

    model = app.state.model
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(fare=float(y_pred))
    # $CHA_END


# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
#####app.state.model = load_model()
# $WIPE_END

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
#delay_pipeline.score(X_test, y_test)
