import csv


def get_trained_data_from_file():
    """Load the theta values from the file."""
    try:
        with open("trained_data.csv", "r") as file:
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
        return 0, 0


def predict():
    """Predict the price of a car based on its mileage."""
    try:
        mileage = float(input("Enter a mileage: "))
        if mileage < 0:
            raise ValueError
    except ValueError:
        print("The mileage must be a positive number.")
        return
    theta0, theta1, mean_, std_ = get_trained_data_from_file()
    normalized_mileage = (mileage - mean_) / std_
    print("normalized_mileage: {}".format(normalized_mileage))
    price = theta0 + theta1 * normalized_mileage
    print("The predicted price is: [{:.2f}]".format(price))


if __name__ == "__main__":
    predict()