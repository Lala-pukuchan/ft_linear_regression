import numpy as np


class LinearRegressionModel:
    """Linear Regression Model"""

    def __init__(self, theta, alpha=0.01, max_iter=1000):
        """Initialize the model."""
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)

    def gradient(self, x, y):
        """Calculate the gradient of the cost function."""
        m = len(x)
        x_1 = np.insert(x, 0, 1, axis=1)
        return 1 / m * np.dot(x_1.T, np.dot(x_1, self.theta) - y)

    def fit(self, x, y):
        """Fit the model to the training data."""
        for _ in range(self.max_iter):
            if _ % 100 == 0:
                print("Iteration: {}".format(_))
                print("theta: {}".format(self.theta))
            self.theta -= self.alpha * self.gradient(x, y)

    @staticmethod
    def normalize(x):
        """Normalize the data."""
        mean_ = np.mean(x)
        std_ = np.std(x)
        x_norm = (x - mean_) / std_
        return mean_, std_, x_norm


if __name__ == "__main__":
    try:
        # load the data
        data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
        x = data[:, 0].reshape(-1, 1)
        y = data[:, 1].reshape(-1, 1)
        print("x.shape: {}".format(x.shape))
        print("y.shape: {}".format(y.shape))
        print("x: {}".format(x))
        print("y: {}".format(y))

        # normalize the data
        mean_, std_, x_norm = LinearRegressionModel.normalize(x)
        print("x_norm: {}".format(x_norm))

        # initialize and fit the model
        initial_theta = np.zeros((2, 1))
        model = LinearRegressionModel(theta=initial_theta)
        print("theta: {}".format(model.theta))
        model.fit(x_norm, y)
        print("theta: {}".format(model.theta[0][0]))

        # output theta values to file
        with open("trained_data.csv", "w") as file:
            file.write("{},{}\n".format(model.theta[0][0], model.theta[1][0]))
            file.write("{},{}\n".format(mean_, std_))

    except OSError:
        print("The data file is missing.")
        exit(1)
