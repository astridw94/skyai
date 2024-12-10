# Imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

#Initialize FastAPI

app = FastAPI()


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
fare_prediction_pipeline = pickle.load(open("skypkg/model_pickle_files/rnn_model_fares_prediction.pkl"))

print('models loaded')


# Load lookup tables

flight_details_table = pd.read_csv("data/flight-detail-grouped.csv")
flight_details_table.drop(columns='Unnamed: 0')
city_details_table = pd.read_csv("data/output_passengers.csv",encoding='UTF-8')



print('data loaded')



# Delay probability endpoint
@app.get("/delay_proba")
def predict_delay(
        flight_number: int,  # 1960
        departure_month: int # 1
        ):

    print(type(flight_number))
    print(type(departure_month))

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
    flight_detail = flight_details_table.loc[flight_details_table['flight.number'] == flight_number].copy()
    print(flight_detail)
    if flight_detail.empty:
        return {
            "Response": f"No details found for flight number {flight_number}"
        }

    # Add input departure_month as a new column
    flight_detail['departure.month'] = departure_month

    # This is the data that goes into pipeline for prediction
    X_new001 = flight_detail.copy()
    # Predict the probability of delay
    predicted_proba = delay_pipeline.predict_proba(X_new001)[:, 1]
    binary_pred = delay_pipeline.predict(X_new001)
    if binary_pred == 1:
        outcome = "Delayed"
    if binary_pred == 0:
        outcome = "On time"
    print(predicted_proba[0])
    print(type(predicted_proba[0]))

    res={"proba_delay": float(predicted_proba[0]),
    "outcome": str(outcome),
    "bin_classification": int(binary_pred)}

    return res












@app.get("/demand_forecast")
def demand_forecast(
    city_name: object, # Brusque
):

"""
    Return the predicted number of passengers for a given city if it had an airport
    (cities w/o airport) or actual demand (cities w/ airport)

    Parameters:
        flight_number (int): The flight number to search for.
        departure_month (int): The departure month (e.g., 1 for January).
    OBS:
        flight_details_table initialized internally in fast.py

    Returns:
        float: Predicted number of passangers (cities w/o airport)
        float: Actual number of passangers (cities w/ airport)
        float: Count of airports in the city
        float: Count of airports in the immediate metropolitan region

    """



    pass






@app.get("/pricing_forecast")
def pricing_forecast(
    departure_airport: object, #SBBP
    arrival_airport: object, #SBRJ
    prediction_period: int, #3 months
    ):

    """
    Predicts the expected price of a specific route for the next month.

    Parameters:
        route (string): defined as the combination of two airports IATA codes
        prediction_period (int): the length of the

    Returns:
        float: The predicted prices of this specific route next month.

    """
    route = (departure_airport,arrival_airport)

    print(flight_detail)
    if flight_detail.empty:
        return {
            "Response": f"No details found for flight number {flight_number}"
        }
    return

### Input

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello. Congrats, the API is working, go check docs")
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
