import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionModel:
    """Linear Regression Model"""

    def __init__(self, theta, alpha=0.01, max_iter=1001):
        """Initialize the model."""
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)

    def gradient(self, x, y):
        """Calculate the gradient of the cost function."""
        m = len(x)
        x_1 = np.insert(x, 0, 1, axis=1)
        return 1 / m * np.dot(x_1.T, self.predict(x) - y)

    def fit(self, x, y):
        """Fit the model to the training data."""
        mses = []
        for _ in range(self.max_iter):
            if _ % 200 == 0:
                print("Iteration: {}".format(_))
                print("theta: {}".format(self.theta))
                self.plot_actual_vs_predicted_data(x, y, model.predict(x), _)
            mses.append(np.mean((model.predict(x) - y) ** 2))
            self.theta -= self.alpha * self.gradient(x, y)

        # plot MSE vs iterations
        plt.figure()
        plt.plot(range(self.max_iter), mses)
        plt.title("MSE vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.savefig("mse_vs_iterations.png")
        plt.show()

    @staticmethod
    def normalize(x):
        """Normalize the data."""
        mean_ = np.mean(x)
        std_ = np.std(x)
        x_norm = (x - mean_) / std_
        return mean_, std_, x_norm

    def predict(self, x):
        """Predict data."""
        x_1 = np.insert(x, 0, 1, axis=1)
        return np.dot(x_1, self.theta)

    def plot_actual_vs_predicted_data(self, x, y, y_pred, iteration=None):
        plt.figure()
        plt.scatter(x, y, color="blue")
        plt.plot(x, y_pred, color="red")
        for xi, yi, y_pred_i in zip(x, y, y_pred):
            plt.plot([xi, xi], [yi, y_pred_i], color="green", linestyle="dotted")
        plt.title("Actual Data")
        plt.xlabel("km")
        plt.ylabel("price")
        if iteration is not None:
            plt.text(
                0.6,
                0.95,
                f"Iteration: {iteration}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=plt.gca().transAxes,
            )
            file_name = f"actual_vs_predicted_data_{iteration}.png"
        else:
            file_name = "actual_vs_predicted_data.png"
        plt.text(
            0.6,
            1.5,
            f"Theta: {self.theta[0][0]:.2f}, {self.theta[1][0]:.2f}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
        )
        plt.savefig(file_name)
        plt.show()

    @staticmethod
    def get_cost(x, y, theta_pair):
        """Calculate the cost."""
        m = len(y)
        x_1 = np.insert(x, 0, 1, axis=1)
        pred = np.dot(x_1, theta_pair)
        return (1 / 2 * m) * np.sum((pred - y) ** 2)

    @staticmethod
    def plot_cost_function_landscape(x_norm):
        """Plot the cost function landscape."""
        # create a grid of theta values
        theta0 = np.linspace(5000, 7000, 100)
        theta1 = np.linspace(-1200, -1000, 100)
        theta0_grid, theta1_grid = np.meshgrid(theta0, theta1)

        # calculate cost for each theta combination
        costs = np.zeros_like(theta0_grid)
        for i in range(theta0_grid.shape[0]):
            for j in range(theta0_grid.shape[1]):
                theta_pair = np.array([[theta0_grid[i, j]], [theta1_grid[i, j]]])
                costs[i, j] = LinearRegressionModel.get_cost(x_norm, y, theta_pair)

        # plot the cost function landscape
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(theta0_grid, theta1_grid, costs, cmap="viridis")
        ax.set_xlabel("Theta0")
        ax.set_ylabel("Theta1")
        ax.set_zlabel("Cost")
        ax.set_title("Cost Function Landscape")
        plt.savefig("cost_function_landscape.png")
        plt.show()

    @staticmethod
    def plot_actual_vs_normalized_data(x, y, x_norm, mean_, std_):
        """Plot actual vs normalized data."""
        plt.figure(figsize=(12, 6))

        # plot actual data
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, color="blue")
        plt.title("Actual Data")
        plt.xlabel("km")
        plt.ylabel("price")
        plt.text(
            0.6,
            0.95,
            f"Mean: {mean_:.2f}\nStd: {std_:.2f}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
        )

        # plot normalized data
        plt.subplot(1, 2, 2)
        plt.scatter(x_norm, y, color="green")
        plt.title("Normalized Data")
        plt.xlabel("normalized km")
        plt.ylabel("price")
        plt.savefig("actual_vs_normalized_data.png")
        plt.show()


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

        # plot actual vs normalized data
        LinearRegressionModel.plot_actual_vs_normalized_data(x, y, x_norm, mean_, std_)

        # plot the cost function landscape
        LinearRegressionModel.plot_cost_function_landscape(x_norm)

        # initialize and fit the model
        initial_theta = np.zeros((2, 1))
        model = LinearRegressionModel(theta=initial_theta)
        model.plot_actual_vs_predicted_data(x, y, model.predict(x_norm))
        print("theta: {}".format(model.theta))
        model.fit(x_norm, y)
        model.plot_actual_vs_predicted_data(x, y, model.predict(x_norm))
        print("theta: {}".format(model.theta[0][0]))

        # calculate the MSE of the model
        mse = LinearRegressionModel.get_cost(x_norm, y, model.theta)
        print(
            "MSE of this model:", mse
        )
        print(
            "RMSE of this model:", np.sqrt(mse)
        )

        # output theta values to file
        with open("trained_data.csv", "w") as file:
            file.write("{},{}\n".format(model.theta[0][0], model.theta[1][0]))
            file.write("{},{}\n".format(mean_, std_))

    except OSError:
        print("The data file is missing.")
        exit(1)
