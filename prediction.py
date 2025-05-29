from model import UnivariteLinearRegression
from loguru import logger

PARAMS_PATH = "./linreg_params.json"


if __name__ == "__main__":

    model = UnivariteLinearRegression()
    model.load_params(PARAMS_PATH)

    print()
    print("============PREDICTION PROGRAM===============")
    mileage = float(input("Enter Mileage of Car (KM)>>>"))

    price_estimate = model.predict(mileage)

    print(f"Price estimate: {price_estimate}")

    


