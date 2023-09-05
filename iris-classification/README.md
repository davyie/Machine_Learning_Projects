# iris-classification
This repository is an exploration project about deep learning applied to classification. 

# Dataset
We are working with the Iris dataset which contains data about iris flowers. The features are the length and width of petals and sepals. The unique widths and lengths determine what type of flower it is. There are three types of flowers in the iris dataset. 

The three types of iris are, 
- Iris-setosa
- Iris-versicolor
- Iris-virginica

The dataset contains 150 entries. The mean length of sepal is `5.84` cm and the mean width is `3.1` cm. The mean length of petal is `3.75` and the mean width is `0.76` cm. 

The standard deviation of the length of sepal is `0.82` and for the width it is `0.44`. Moreover, the standard deviation of length of petal is `1.76` and for the width it is `0.76`. 
The standard deviation tells how the dispersed the lengths and widths are from the mean. For example, one standarad deviation might include `60%` of the flowers which have sepal length between `5.84` and `5.84 + 0.83 = 6.67`. If we increase the number of standard deviations we will include more data points in our interval. 

# Model 
The chosen model for this task is a feedforward neural network with relu activation function. 
We have a hidden layer of size `32` with an output layer of size `3` that outputs logits. If we want to output probabilities we have to append a softmax layer at the end. 

However, we can use the logits because they are correlated with probabilities. The higher the logit the higher the probability. Negative logit indicates probability less than 1/2. 
To perform inference we can perform argument max on the logits to find the index with the highest logit which equals the predicted class. 

The optimizer is Stochastic Gradient Descent with learning rate $1e-3$. 
We use Cross entropy loss function. 
The model can be trained on multiple epochs and batch sizes. 

The model achieves the highest accuracy of $0.96$ with epoch equals 1000 and batch size equals 32. But we perform inference on the whole dataset so the model has overfitted with those hyperparameters. 
