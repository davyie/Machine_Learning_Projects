from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from dataset import Dataset

def main():
    print("Hello World!")
    path = "../dataset/housing.data"
    dataset = Dataset(path)
    lm = LinearRegression()

    lm.fit(dataset.X, dataset.Y)

    y_hats = lm.predict(dataset.X)

    print(y_hats[0], dataset.Y[0])
    mse = mean_squared_error(dataset.Y, y_hats)
    print("Linear Regression - MSE: %f" % mse)

    enm = ElasticNet()
    enm.fit(dataset.X, dataset.Y)

    y_hats = enm.predict(dataset.X)

    print(y_hats[0], dataset.Y[0])
    mse = mean_squared_error(dataset.Y, y_hats)
    print("ElasticNet - MSE: %f" % mse)

if __name__ == "__main__":
    main()