import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

def plot_fig(df):
    fig1 = df.plot.line(x='year', y='days')
    fig1.set_xlabel("Year")
    fig1.set_ylabel("Number of frozen days")
    fig1.get_figure().savefig("plot.png")

def fetch_year_and_days_np_arr(df):
    df['constant_dimension'] = 1
    days_np = df['days'].to_numpy()
    year_np = df[['constant_dimension','year']].to_numpy(dtype=int)
    return days_np, year_np

if __name__ == "__main__":
    filename = sys.argv[1]
    B1_sign = ""

    ## todo fill in the answers
    answer_to_5b = """In the linear regression model, the signs of Beta1 here indicate the trend of the slope. Here, Beta1 has a negative slope (which would be indicated as <), which makes the value (number of days of ice) to follow a negative trend. This reduces its corresponding value from Beta0, which is positive."""
    answer_to_6b = """Yes, x_star does seem as a compelling prediction based on the given data and graph. The graph shows a negative trend with respect to the number of frozen days. This is in sync with the answer in 5b as well. So, the prediction where number of frozen days goes down to 0 seems to be accurate because of its negative trend in the plot"""
    complete_df = pd.read_csv(filename, index_col=0).reset_index()
    plot_fig(df=complete_df)
    print("Q3a:")
    days, year = fetch_year_and_days_np_arr(df=complete_df)
    print(year)
    print("Q3b:")
    print(days)
    print("Q3c:")
    xTx = np.dot(np.transpose(year), year)
    print(xTx)
    print("Q3d:")
    inv = np.linalg.inv(xTx)
    print(inv)
    print("Q3e:")
    PInv = np.dot(inv, np.transpose(year))
    print(PInv)
    print("Q3f:")
    B = np.dot(PInv, days)
    print(B)
    predicted_value = B[0] + B[1]*2021
    print(f"Q4: {predicted_value}")
    if B[1] < 0:
        B1_sign = "<"
    elif B[1] > 0:
        B1_sign = ">"
    else:
        B1_sign = "="
    print(f"Q5a: {B1_sign}")
    print(f"Q5b: {answer_to_5b}")
    solve_x = -(B[0]/B[1])
    print(f"Q6a: {solve_x}")
    print(f"Q6b: {answer_to_6b}")

