from dataloader import DataLoader 
from visualizer import Visualizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor

def main():
  path = 'archive/train.csv'
  dl = DataLoader()
  dl.load_data(path)
  visualizer = Visualizer()

  def general_info():
    print(dl.raw_data.info())

  def is_duplicates():
    print(dl.raw_data.duplicated().any())

  def get_unique_counts():
    print(dl.raw_data.apply(lambda x: len(x.unique())))

  # Check what features exists 
  def type_of_features():
    print('Categorical: ', dl.categorical_features())
    print('Numerical: ', dl.numerical_features())

  # Count the number of data points with each feature 
  def count_dp_per_feature():
    categories = dl.categorical_features()
    for c in categories:
      print('-----------------')
      print('Feature: ', c)
      print('-----------------')
      print(dl.raw_data[c].value_counts())
      print('\n')
  
  def visualize(feature_name):
    labels = list(dl.data[feature_name].unique())
    visualizer.visualize_plot(feature_name, dl.X, labels)
  
  # visualize('item_type') ## Visualize item_type

  def get_data():
    X, Y = dl.get_X_Y()
    return X, Y

  def train_and_evaluate(model_name, model):
    X, Y = get_data()
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(StandardScaler(), model)

    pipeline.fit(X_train, Y_train)

    y_hat = pipeline.predict(X_validate)

    rmse = np.sqrt(mean_squared_error(Y_validate, y_hat))
    model_score = r2_score(Y_validate, y_hat)

    print('----------------------------------')
    print(model_name, ' Report:')
    print('----------------------------------')
    print('RMSE: ', rmse)
    print('R2 Score: ', model_score)

  def run():
    # Linear Regression 
    model = SGDRegressor(loss='squared_error')
    train_and_evaluate('SGD', model)

  run()


if __name__ == '__main__':
    main()