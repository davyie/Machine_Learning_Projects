# bigmart-prediction-sales
This repository is a machine learning project to predict sales for products. 

# Purpose 
I want to apply my knowledge to a practical problem. Therefore, in an attempt to achieve my objective I have decide to explore the dataset by BigMart. 

Why? To gain an understanding of how products sales or being able to predict how product sales is important. One of the reasons is that we get an understanding of how the market is moving for different products. This helps us make decisions of whether to improve marketing, or discontinue a product or increase goods of the product. This contributes directly to the economy of a business and the decision making. 

# Dataset 
This project revolves around the dataset BigMart. The dataset contains two files, `train.csv` and `test.csv`. The training set is located in `train.csv` and contains 8523 data points where each data point has 10 features. The features are split into numerical and categorical where as 4 of them are numerical and the rest are categorical. 

## Explore the data 
The features are the following, 

- `Item identifier`: Categorical, a unique id 
- `Item_Weight`: Numerical, the weight of the item 
- `Item_Fat_Content`: Categorical, either low fat or regular fat. This exists for eatable items 
- `Item_Visibility`: Numerical, the % of total display area of all products in a store allocated to the item 
- `Item_Type`: Category, the type of item 
- `Item_MRP`: Numerical, the maximum price of the item 
- `Outlet_Identifier`: Categorical, unique store id 
- `Outlet_Establishment_Year` : Numerical, the year the item was established 
- `Outlet_Size`: Categorical, the size of store 
- `Outlet_Location_Type`: Categorical, the type of city the store is located for the item
- `Outlet_Type`: Categorical, the type of store, supermarket or grocery store 

The `target` is a value called `Item_Outlet_Sales` which is the total sales in `dollar` of the item for the store. 

![Alt text][assets/general_info_bigmart.png]

The image displays the different features and how many data points have a non-null value for each feature. A close observation show that `outlet_size` has the least number of data points with non-null values. This indicates we have to fill in the data points with a `null` in `outlet-size`. 

We can also display duplicates with `pandas` by calling the `duplicates()` with `DataFrame` object. 

## Clean the data 
Now that we have explored the data a bit we are going to clean it before we can use it. 
We make the feature identifiers lowercase first. Then we have to fill in the `null` values for the features of the data points. 

To fill in the `item-weight` we compute the `average` of all the items. The positive about this method is that we get a value from it. The negative is that it might not be accurate representative for the item. However, it is one out of ten features the influence might not be that great. 

The second feature we have to fill in is `outlet-size`. For this we use the mode, i.e., we think the item is sold in the most common type of store. 

Third feature we need to modify is the `item_fat_content`. This value has four different values but they can be reduced to two values, `low fat` or `regular`. We do this with `replace({'low fat': ..., 'LF': ..., .. })`.

Now we want to add two more features `outlet_years` which indicates how many years the outlet has been established since its start and `item_category` which indicates whether the item is a food, drink or non-consumable. 

The last feature we might want to modify is `item_visibility` because there are several items which have visibility `0`. To modify it we have to compute the average visibility of all items and replace 0 with the average. 

## Visualize the data 
To visualize the data we can use `matplotlib` and `seaborn`. `Matplotlib` is the underlying library to draw figures and `seaborn` is a high level version which does not require that much code compare to `matplotlib`. Therefore, it is more nice to use `seaborn`. 

## Preprocessing 
We have three fields which are identifiers, and those three fields are `item_identifier`, `item_type` and `outlet_identifier`. We can use `sklearn.preprocessing.LabelEncoder` to encode those into values. 
Now we have to add `one-hot` encoding to some features. This is will add new features which replaces the old ones. The features we are going to `one-hot` encode are `item_fat_content`, `outlet_size`, `outlet_location_type`, `outlet_type` and `item_category`. 
We use `pandas.get_dummies()` to obtain the features. 
Then we are ready to put it in to our models. We are going to drop `outlet_establishment_year` due to it not being relevant to prediction and `item_outlet_sales` because it is the target we want to predict. 

Now we are ready to apply machine learning models! 

## Machine learning 
To actually make use of the data we have to build a pipeline and split the data into training and validation test set. We use `sklearn.model_selection.train_test_split` to do this operation. Then we create a pipeline with `sklearn.pipeline.make_pipeline` to streamline the training by normalizing and inputting the data to the model. Then we fit the model and compute `y hat` for `X_validation`. Then we compute the `RMSE - Root Mean Square Error` and `R2 score` to evaluate the error and model. 

We have explore two models which are `SGDRegressor` and `LinearRegression`, and it seems like `LinearRegression` obtains higher `RMSE` and `R2 score` of the two models. 

## Improvements 
