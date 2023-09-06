# Boston Housing Sales Prediction 
This project is about predicting housing sale median prices in 1000$. We have multiple features that we are going to explore and see if correlations exist in the data. 

## Dataset 
We are using Boston Housing Sales data. From the attribute information we can observe that the data has 14 features where the last one is the value we want to predict. All of the values are numerical and one of them is boolean. 
This indicates we are going to perform regression. 

Here are the features: 
- CRIM: crime rate per capita by town 
- ZN: the proportion of residential land zoned for lots over 25, 000 sq.ft
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable, equals 1 if tract bounds river; 0 else. 
- NOX: Nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling 
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town 
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
- LSTAT: % lower status of the population 
- MEDV: Median value of owner-occupied homes in $1000's (Target)

## Pearson Correlation Coefficient 

The first statistical analysis we are going to perform is Pearson Correlation Coefficient which tells us how the features relate to each other. 

Why does it matter? To gain an understanding of correlation we can infer missing values from other variables because two variable might have a linear relationship. Correlation also helps us understand causal relationships between variables. We can also use this analysis to reduce the dimension and improve computation time and performance of our machine learning model. 

![Alt text](./assets//PCC_matrix.png?raw=true "Title")

From the matrix above we can observe that INDUS and TAX has quite high correlation. This means that when a house is located near land with non-retail business acres its taxes gets affected. For example, if the acres increase then the tax increase aswell. 

## F-test & Linear Regression 
This is a statistical tool to model the relationship between dependent and independent variable. F-tests are used to determine if an independent variable has a relationship with dependent variable that is statistically significant, i.e., contributes to the dependent variable. 

To perform this test between a feature and a target we have to create a linear regression model with the form $Y = \beta_0 + \beta_1 X + \epsilon$. 

The null hypothesis is $\beta_1 = 0$ and we discard it for the hypothesis $\beta_1 \neq 0$. 

When we have fitted our linear regression model we have to predict all the $Y_{hat}$ which is from the linear line. Then we compute Sum of Square Error SSE, Corrected Sum of Squares for Model SSM, Corrected Degress Of Freedom for Model DFM and Degrees of Freedom for Error DFE. 

This is how they are computed: 

$SSM = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^{2}$ 

$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^{2}$

$DFM = p -1$

$DFE = n - p$ 

where $\hat{y}$ is the predicted value using linear regression, $\bar{y}$ is the sample arithmetic mean, $p$ is the number of parameters and $n$ is the number of samples. 

Once all of them have been computed we can we can compute $F$ value which is has the following formula, 

$F = (SSM/DFM) / (SSE/DFE)$

To draw conclusions we need the confidence interval which can be found by computing the value of F-distribution on the significant level $\alpha$ or confidence level $100 - \alpha$ using $DFE$ and $DFM$ as degrees of freedom. 
And if F is within the confidence interval we accept the null hypothesis and reject it otherwise. 

The confidence interval for our datasat is $[0, 3]$ and when we computed F test for all features with target we observe that we can reject the hypothesis for all. The means that the coefficient explains the linear relationship between feature and target. 




