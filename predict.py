import csv
import matplotlib.pyplot as plt
import os

file_path = "./trained_data.csv"


def get_trained_data_from_file():
    """Load the theta values from the file."""
    try:
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            row0 = next(csv_reader)
            theta0 = float(row0[0])
            theta1 = float(row0[1])
            print("theta0: {}".format(theta0))
            print("theta1: {}".format(theta1))
            row1 = next(csv_reader)
            mean_ = float(row1[0])
            std_ = float(row1[1])
            return theta0, theta1, mean_, std_
    except FileNotFoundError:
        exit(1)


def predict():
    """Predict the price of a car based on its mileage."""
    try:
        mileage = float(input("Enter a mileage: "))
        if mileage < 0:
            raise ValueError
    except ValueError:
        print("The mileage must be a positive number.")
        return
    if os.path.exists(file_path):
        theta0, theta1, mean_, std_ = get_trained_data_from_file()
        normalized_mileage = (mileage - mean_) / std_
        print("normalized_mileage: {}".format(normalized_mileage))
        price = theta0 + theta1 * normalized_mileage
        print("The predicted price is: [{:.2f}]".format(price))
    else:
        theta0 = 0.0
        theta1 = 0.0
        price = theta0 + theta1 * mileage
        print("The predicted price is: [{:.2f}]".format(price))


if __name__ == "__main__":
    predict()
