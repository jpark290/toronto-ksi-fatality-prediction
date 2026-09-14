"""Shared schema for the deployed KSI fatal-collision model.

This matches the final modelling feature set in the supplied notebook:
HOOD_158 and NEIGHBOURHOOD_158 are deliberately excluded.
"""

RANDOM_STATE = 42
TARGET_COL = "TARGET_FATAL"

NUMERIC_FEATURES = [
    "LATITUDE", "LONGITUDE", "YEAR", "MONTH", "DAY_OF_WEEK", "HOUR"
]

BINARY_FEATURES = [
    "DISABILITY", "ALCOHOL", "TRSN_CITY_VEH", "TRUCK", "REDLIGHT",
    "MOTORCYCLE", "SPEEDING", "CYCLIST", "PASSENGER", "PEDESTRIAN",
    "AG_DRIV", "AUTOMOBILE",
]

MODE_FEATURES = [
    "ROAD_CLASS", "DISTRICT", "TRAFFCTL", "VISIBILITY", "RDSFCOND", "LIGHT"
]

UNKNOWN_FEATURES = ["INITDIR", "ACCLOC", "DIVISION"]

CATEGORICAL_FEATURES = BINARY_FEATURES + MODE_FEATURES + UNKNOWN_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Winning model from the executed notebook output.
BEST_SVM_PARAMS = {"C": 10.0, "kernel": "linear"}
