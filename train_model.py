import pandas as pd
from model import UnivariteLinearRegression
import json

if __name__ == "__main__":
    
    DATA_PATH = "./data.csv"
    LR=0.01
    ITERATIONS=10000

    linreg = UnivariteLinearRegression()

    df = pd.read_csv(DATA_PATH)
    x = df["km"].tolist()
    y = df["price"].tolist()

    linreg.fit(x, y, n_iterations=ITERATIONS, lr=LR)
    linreg.dump_params("linreg_params.json")



