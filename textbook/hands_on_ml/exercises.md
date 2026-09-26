# All Exercises

## Chapter 2

The following exercises are based on this chapter’s housing dataset:
1. Try a support vector machine regressor (sklearn.svm.SVR)
with various hyperparameters, such as kernel="linear" (with
various values for the C hyperparameter) or kernel="rbf"
(with various values for the C and gamma hyperparameters).
Note that support vector machines don’t scale well to large
datasets, so you should probably train your model on just the
first 5,000 instances of the training set and use only 3-fold
cross-validation, or else it will take hours. Don’t worry about
what the hyperparameters mean for now; these are explained
in the online chapter on SVMs at https://homl.info/. How does
the best SVR predictor perform?
2. Try replacing the GridSearchCV with a RandomizedSearchCV.
3. Try adding a SelectFromModel transformer in the preparation
pipeline to select only the most important attributes.
4. Try creating a custom transformer that trains a k-nearest
neighbors regressor
(sklearn.neighbors.KNeighborsRegressor) in its fit()
method, and outputs the model’s predictions in its transform()
method. The KNN regressor should use only the latitude and
longitude as input and predict the median income. Next, add
this new transformer to the preprocessing pipeline. This will
add a feature representing the smoothed median income over
the nearby districts.
5. Automatically explore some preparation options using
RandomizedSearchCV.
6. Try to implement the StandardScalerClone class again from
scratch, then add support for the inverse_transform()
method: executing scaler.​
inverse_transform(scaler.fit_transform(X)) should return
an array very close to X. Then add support for feature names:
set feature_names_in_ in the fit() method if the input is a
DataFrame. This attribute should be a NumPy array of column
names. Lastly, implement the get_feature_names_out()method: it should have one optional input_features=None
argument. If passed, the method should check that its length
matches n_features_in_, and it should match
feature_names_in_ if it is defined; then input_features
should be returned. If input_features is None, then the
method should either return feature_names_in_ if it is defined
or np.array(["x0", "x1", ...]) with length n_features_in_
otherwise.
7. Tackle a regression task of your choice by following the process
you learned in this chapter. For example, you can try tackling
the Vehicle dataset, where the goal is to predict the selling
price of a used car, based on its age, the number of kilometers
it has driven, its make and model, and more. Another good
dataset to try is the Bike Sharing dataset: the objective is to
predict the number of bikes rented within a period of time
(column cnt), based on the day of the week, the time, and the
weather conditions.
Solutions to these exercises are available at the end of this chapter’s
notebook, at https://homl.info/colab-p.

## Chapter 3

1. Try to build a classifier for the MNIST dataset that
achieves over 97% accuracy on the test set. Hint:
the KNeighborsClassifier works quite well for this
task; you just need to find good hyperparameter
values (try a grid search on the weights and
n_neighbors hyperparameters).
2. Write a function that can shift an MNIST image in
any direction (left, right, up, or down) by one pixel.⁠7
Then, for each image in the training set, create four
shifted copies (one per direction) and add them to
the training set. Finally, train your best model on
this expanded training set and measure its accuracy
on the test set. You should observe that your model
performs even better now! This technique of
artificially growing the training set is called data
augmentation or training set expansion.
3. Tackle the Titanic dataset. A great place to start is
on Kaggle. Alternatively, you can download the data
from https://homl.info/titanic.tgz and unzip this
tarball like you did for the housing data in
Chapter 2. This will give you two CSV files, train.csv
and test.csv, which you can load usingpandas.read_csv(). The goal is to train a classifier
that can predict the Survived column based on the
other columns.
4. Build a spam classifier (a more challenging
exercise):
a. Download examples of spam and ham from
Apache SpamAssassin’s public datasets.
b. Unzip the datasets and familiarize yourself
with the data format.
c. Split the data into a training set and a test
set.
d. Write a data preparation pipeline to convert
each email into a feature vector. Your
preparation pipeline should transform an
email into a (sparse) vector that indicates the
presence or absence of each possible word.
For example, if all emails only ever contain
four words, “Hello”, “how”, “are”, “you”, then
the email “Hello you Hello Hello you” would
be converted into a vector [1, 0, 0, 1]
(meaning [“Hello” is present, “how” is absent,
“are” is absent, “you” is present]), or [3, 0, 0,
2] if you prefer to count the number of
occurrences of each word.
You may want to add hyperparameters to
your preparation pipeline to control whether
to strip off email headers, convert each email
to lowercase, remove punctuation, replace all
URLs with “URL”, replace all numbers with
“NUMBER”, or even perform stemming (i.e.,trim off word endings; there are Python
libraries available to do this).
e. Finally, try out several classifiers and see if
you can build a great spam classifier, with
both high recall and high precision.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 4

1. Which linear regression training algorithm can you use if you
have a training set with millions of features?
2. Suppose the features in your training set have very different
scales. Which algorithms might suffer from this, and how? What
can you do about it?
3. Can gradient descent get stuck in a local minimum when
training a logistic regression model?
4. Do all gradient descent algorithms lead to the same model,
provided you let them run long enough?
5. Suppose you use batch gradient descent and you plot the
validation error at every epoch. If you notice that the validation
error consistently goes up, what is likely going on? How can
you fix this?
6. Is it a good idea to stop mini-batch gradient descent
immediately when the validation error goes up?
7. Which gradient descent algorithm (among those we discussed)
will reach the vicinity of the optimal solution the fastest? Which
will actually converge? How can you make the others converge
as well?
8. Suppose you are using polynomial regression. You plot the
learning curves and you notice that there is a large gap
between the training error and the validation error. What is
happening? What are three ways to solve this?
9. Suppose you are using ridge regression and you notice that the
training error and the validation error are almost equal and
fairly high. Would you say that the model suffers from high biasor high variance? Should you increase the regularization
hyperparameter α or reduce it?
10. Why would you want to use:
a. Ridge regression instead of plain linear regression (i.e.,
without any regularization)?
b. Lasso instead of ridge regression?
c. Elastic net instead of lasso regression?
11. Suppose you want to classify pictures as outdoor/indoor and
daytime/nighttime. Should you implement two logistic
regression classifiers or one softmax regression classifier?
12. Implement batch gradient descent with early stopping for
softmax regression without using Scikit-Learn, only NumPy.
Use it on a classification task such as the iris dataset.
Solutions to these exercises are available at the end of this chapter’s
notebook, at https://homl.info/colab-p.

## Chapter 5

1. What is the approximate depth of a decision tree trained (without
restrictions) on a training set with one million instances?
2. Is a node’s Gini impurity generally lower or higher than its parent’s? Is
it generally lower/higher, or always lower/higher?
3. If a decision tree is overfitting the training set, is it a good idea to try
decreasing max_depth?
4. If a decision tree is underfitting the training set, is it a good idea to try
scaling the input features?
5. If it takes one hour to train a decision tree on a training set containing
one million instances, roughly how much time will it take to train
another decision tree on a training set containing ten million instances?
Hint: consider the CART algorithm’s computational complexity.
6. If it takes one hour to train a decision tree on a given training set,
roughly how much time will it take if you double the number of
features?
7. Train and fine-tune a decision tree for the moons dataset by following
these steps:
a. Use make_moons(n_samples=10000, noise=0.4) to generate a
moons dataset.
b. Use train_test_split() to split the dataset into a training set
and a test set.
c. Use grid search with cross-validation (with the help of the
GridSearchCV class) to find good hyperparameter values for a
DecisionTreeClassifier. Hint: try various values for
max_leaf_nodes.
d. Train it on the full training set using these hyperparameters, and
measure your model’s performance on the test set. You should
get roughly 85% to 87% accuracy.
8. Grow a forest by following these steps:
a. Continuing the previous exercise, generate 1,000 subsets of the
training set, each containing 100 instances selected randomly.
Hint: you can use Scikit-Learn’s ShuffleSplit class for this.
b. Train one decision tree on each subset, using the best
hyperparameter values found in the previous exercise. Evaluate
these 1,000 decision trees on the test set. Since they were
trained on smaller sets, these decision trees will likely performworse than the first decision tree, achieving only about 80%
accuracy.
c. Now comes the magic. For each test set instance, generate the
predictions of the 1,000 decision trees, and keep only the most
frequent prediction (you can use SciPy’s mode() function for
this). This approach gives you majority-vote predictions over the
test set.
d. Evaluate these predictions on the test set: you should obtain a
slightly higher accuracy than your first model (about 0.5 to 1.5%
higher). Congratulations, you have trained a random forest
classifier!
Solutions to these exercises are available at the end of this chapter’s
notebook, at https://homl.info/colab-p.

## Chapter 6

1. If you have trained five different models on the exact same training data,
and they all achieve 95% precision, is there any chance that you can
combine these models to get better results? If so, how? If not, why?
2. What is the difference between hard and soft voting classifiers?
3. Is it possible to speed up training of a bagging ensemble by distributing
it across multiple servers? What about pasting ensembles, boosting
ensembles, random forests, or stacking ensembles?
4. What is the benefit of out-of-bag evaluation?
5. What makes extra-trees ensembles more random than regular random
forests? How can this extra randomness help? Are extra-trees classifiers
slower or faster than regular random forests?
6. If your AdaBoost ensemble underfits the training data, which
hyperparameters should you tweak, and how?
7. If your gradient boosting ensemble overfits the training set, should you
increase or decrease the learning rate?
8. Load the MNIST dataset (introduced in Chapter 3), and split it into a
training set, a validation set, and a test set (e.g., use 50,000 instances for
training, 10,000 for validation, and 10,000 for testing). Then trainvarious classifiers, such as a random forest classifier, an extra-trees
classifier, and an SVM classifier. Next, try to combine them into an
ensemble that outperforms each individual classifier on the validation
set, using soft or hard voting. Once you have found one, try it on the test
set. How much better does it perform compared to the individual
classifiers?
9. Run the individual classifiers from the previous exercise to make
predictions on the validation set, and create a new training set with the
resulting predictions: each training instance is a vector containing the
set of predictions from all your classifiers for an image, and the target is
the image’s class. Train a classifier on this new training set.
Congratulations—you have just trained a blender, and together with the
classifiers it forms a stacking ensemble! Now evaluate the ensemble on
the test set. For each image in the test set, make predictions with all
your classifiers, then feed the predictions to the blender to get the
ensemble’s predictions. How does it compare to the voting classifier you
trained earlier? Now try again using a StackingClassifier instead. Do
you get better performance? If so, why?
Solutions to these exercises are available at the end of this chapter’s notebook,
at https://homl.info/colab-p.

## Chapter 7

1. What are the main motivations for reducing a dataset’s
dimensionality? What are the main drawbacks?
2. What is the curse of dimensionality?
3. Once a dataset’s dimensionality has been reduced, is it
possible to reverse the operation? If so, how? If not, why?
4. Can PCA be used to reduce the dimensionality of a highly
nonlinear dataset?
5. Suppose you perform PCA on a 1,000-dimensional dataset,
setting the explained variance ratio to 95%. How many
dimensions will the resulting dataset have?
6. In what cases would you use regular PCA, incremental PCA,
randomized PCA, or random projection?
7. How can you evaluate the performance of a dimensionality
reduction algorithm on your dataset?
8. Does it make any sense to chain two different dimensionality
reduction algorithms?
9. Load the MNIST dataset (introduced in Chapter 3) and split
it into a training set and a test set (take the first 60,000
instances for training, and the remaining 10,000 for
testing). Train a random forest classifier on the dataset and
time how long it takes, then evaluate the resulting model on
the test set. Next, use PCA to reduce the dataset’s
dimensionality, with an explained variance ratio of 95%.
Train a new random forest classifier on the reduced dataset
and see how long it takes. Was training much faster? Next,
evaluate the classifier on the test set. How does it compare
to the previous classifier? Try again with an SGDClassifier.
How much does PCA help now?
10. Use t-SNE to reduce the first 5,000 images of the MNIST
dataset down to 2 dimensions and plot the result using
Matplotlib. You can use a scatterplot using 10 different
colors to represent each image’s target class. Alternatively,you can replace each dot in the scatterplot with the
corresponding instance’s class (a digit from 0 to 9), or even
plot scaled-down versions of the digit images themselves (if
you plot all digits the visualization will be too cluttered, so
you should either draw a random sample or plot an instance
only if no other instance has already been plotted at a close
distance). You should get a nice visualization with well-
separated clusters of digits. Try using other dimensionality
reduction algorithms, such as PCA, LLE, or MDS, and
compare the resulting visualizations.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 8

1. How would you define clustering? Can you name a
few clustering algorithms?
2. What are some of the main applications of clustering
algorithms?
3. Describe two techniques to select the right number
of clusters when using k-means.
4. What is label propagation? Why would you
implement it, and how?
5. Can you name two clustering algorithms that can
scale to large datasets? And two that look for
regions of high density?
6. Can you think of a use case where active learning
would be useful? How would you implement it?
7. What is the difference between anomaly detection
and novelty detection?
8. What is a Gaussian mixture? What tasks can you use
it for?
9. Can you name two techniques to find the right
number of clusters when using a Gaussian mixture
model?
10. The classic Olivetti faces dataset contains 400
grayscale 64 × 64–pixel images of faces. Each image
is flattened to a 1D vector of size 4,096. Forty
different people were photographed (10 times each),
and the usual task is to train a model that can
predict which person is represented in each picture.
Load the dataset using thesklearn.datasets.fetch_olivetti_faces()
function, then split it into a training set, a validation
set, and a test set (note that the dataset is already
scaled between 0 and 1). Since the dataset is quite
small, you will probably want to use stratified
sampling to ensure that there are the same number
of images per person in each set. Next, cluster the
images using k-means, and ensure that you have a
good number of clusters (using one of the
techniques discussed in this chapter). Visualize the
clusters: do you see similar faces in each cluster?
11. Continuing with the Olivetti faces dataset, train a
classifier to predict which person is represented in
each picture, and evaluate it on the validation set.
Next, use k-means as a dimensionality reduction
tool, and train a classifier on the reduced set. Search
for the number of clusters that allows the classifier
to get the best performance: what performance can
you reach? What if you append the features from the
reduced set to the original features (again,
searching for the best number of clusters)?
12. Train a Gaussian mixture model on the Olivetti faces
dataset. To speed up the algorithm, you should
probably reduce the dataset’s dimensionality (e.g.,
use PCA, preserving 99% of the variance). Use the
model to generate some new faces (using the
sample() method), and visualize them (if you used
PCA, you will need to use its inverse_transform()
method). Try to modify some images (e.g., rotate,
flip, darken) and see if the model can detect the
anomalies (i.e., compare the output of the
score_samples() method for normal images and for
anomalies).13. Some dimensionality reduction techniques can also
be used for anomaly detection. For example, take
the Olivetti faces dataset and reduce it with PCA,
preserving 99% of the variance. Then compute the
reconstruction error for each image. Next, take
some of the modified images you built in the
previous exercise and look at their reconstruction
error: notice how much larger it is. If you plot a
reconstructed image, you will see why: it tries to
reconstruct a normal face.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 9

1. This neural network playground is a great tool to
build your intuitions without writing any code (it was
built by the TensorFlow team, but there’s nothing
TensorFlow-specific about it; in fact, it doesn’t even
use TensorFlow). In this exercise, you will train
several binary classifiers in just a few clicks, and
tweak the model’s architecture and its
hyperparameters to gain some intuition on how
neural networks work and what their
hyperparameters do. Take some time to explore the
following:a. The patterns learned by a neural net. Try
training the default neural network by
clicking the Run button (top left). Notice how
it quickly finds a good solution for the
classification task. The neurons in the first
hidden layer have learned simple patterns,
while the neurons in the second hidden layer
have learned to combine the simple patterns
of the first hidden layer into more complex
patterns. In general, the more layers there
are, the more complex the patterns can be.
b. Activation functions. Try replacing the tanh
activation function with a ReLU activation
function, and train the network again. Notice
that it finds a solution even faster, but this
time the boundaries are linear. This is due to
the shape of the ReLU function.
c. The risk of local minima. Modify the network
architecture to have just one hidden layer
with three neurons. Train it multiple times (to
reset the network weights, click the Reset
button next to the Play button). Notice that
the training time varies a lot, and sometimes
it even gets stuck in a local minimum.
d. What happens when neural nets are too
small. Remove one neuron to keep just two.
Notice that the neural network is now
incapable of finding a good solution, even if
you try multiple times. The model has too few
parameters and systematically underfits the
training set.e. What happens when neural nets are large
enough. Set the number of neurons to eight,
and train the network several times. Notice
that it is now consistently fast and never gets
stuck. This highlights an important finding in
neural network theory: large neural networks
rarely get stuck in local minima, and even
when they do, these local optima are often
almost as good as the global optimum.
However, they can still get stuck on long
plateaus for a long time.
f. The risk of vanishing gradients in deep
networks. Select the spiral dataset (the
bottom-right dataset under “DATA”), and
change the network architecture to have four
hidden layers with eight neurons each. Notice
that training takes much longer and often
gets stuck on plateaus for long periods of
time. Also notice that the neurons in the
highest layers (on the right) tend to evolve
faster than the neurons in the lowest layers
(on the left). This problem, called the
vanishing gradients problem, can be
alleviated with better weight initialization and
other techniques, better optimizers (such as
AdaGrad or Adam), or batch normalization
(discussed in Chapter 11).
g. Go further. Take an hour or so to play around
with other parameters and get a feel for what
they do to build an intuitive understanding
about neural networks.2. Draw an ANN using the original artificial neurons
(like the ones in Figure 9-3) that computes A ⊕ B
(where ⊕ represents the XOR operation). Hint: A ⊕ B
= (A ∧ ¬ B) ∨ (¬ A ∧ B).
3. Why is it generally preferable to use a logistic
regression classifier rather than a classic perceptron
(i.e., a single layer of threshold logic units trained
using the perceptron training algorithm)? How can
you tweak a perceptron to make it equivalent to a
logistic regression classifier?
4. Why was the sigmoid activation function a key
ingredient in training the first MLPs?
5. Name three popular activation functions. Can you
draw them?
6. Suppose you have an MLP composed of one input
layer with 10 passthrough neurons, followed by one
hidden layer with 50 artificial neurons, and finally
one output layer with 3 artificial neurons. All
artificial neurons use the ReLU activation function.
a. What is the shape of the input matrix X?
b. What are the shapes of the hidden layer’s
weight matrix Wh and bias vector bh?
c. What are the shapes of the output layer’s
weight matrix Wo and bias vector bo?
d. What is the shape of the network’s output
matrix Y?
e. Write the equation that computes the
network’s output matrix Y as a function of X,
Wh, bh, Wo, and bo.7. How many neurons do you need in the output layer
if you want to classify email into spam or ham? What
activation function should you use in the output
layer? If instead you want to tackle MNIST, how
many neurons do you need in the output layer, and
which activation function should you use? What
about for getting your network to predict housing
prices, as in Chapter 2?
8. What is backpropagation and how does it work?
What is the difference between backpropagation and
reverse-mode autodiff?
9. Can you list all the hyperparameters you can tweak
in a basic MLP? If the MLP overfits the training
data, how could you tweak these hyperparameters to
try to solve the problem?
10. Train a deep MLP on the CoverType dataset. You
can load it using
sklearn.datasets.fetch_covtype(). See if you can
get over 93% accuracy on the test set by fine-tuning
the hyperparameters manually and/or using
RandomizedSearchCV.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 10

1. PyTorch is similar to NumPy is many ways, but it
offers some extra features. Can you name the most
important ones?
2. What is the difference between torch.exp() and
torch.exp_(), or between torch.relu() and
torch.relu_()?
3. What are two ways to create a new tensor on the
GPU?
4. What are three ways to perform tensor computations
without using autograd?
5. Will the following code cause a RuntimeError? What
if you replace the second line with z =
t.cos_().exp()? And what if you replace it with z =
t.exp().cos_()?
t = torch.tensor(2.0, requires_grad=True)
z = t.cos().exp_()
z.backward()
How about the following code, will it cause an error?
And what if you replace the third line with w =
v.cos_() * v.sin()? Will w have the same value in
both cases?
u = torch.tensor(2.0, requires_grad=True)
v = u + 1
w = v.cos() * v.sin_()
w.backward()6. Suppose you create a Linear(100, 200) module.
How many neurons does it have? What is the shape
of is weight and bias parameters? What input shape
does it expect? What output shape does it produce?
7. What are the main steps of a PyTorch training loop?
8. Why is it recommended to create the optimizer after
the model is moved to the GPU?
9. What DataLoader options should you generally set
to speed up training when using a GPU?
10. What are the main classification losses provided by
PyTorch, and when should you use each of them?
11. Why is it important to call model.train() before
training and model.eval() before evaluation?
12. What is the difference between torch.jit.trace()
and torch.jit.script()?
13. Use autograd to find the gradient vector of f(x, y) =
sin(x2 y) at the point (x, y) = (1.2, 3.4).
14. Create a custom Dense module that replicates the
functionality of an nn.Linear module followed by an
nn.ReLU module. Try implementing it first using the
nn.Linear and nn.ReLU modules, and then
reimplement it using nn.Parameter and the relu()
function.
15. Build and train a classification MLP on the
CoverType dataset:
a. Load the dataset using
sklearn.datasets.fetch_covtype() andcreate a custom PyTorch Dataset for this
data.
b. Create data loaders for training, validation,
and testing.
c. Build a custom MLP module to tackle this
classification task. You can optionally use the
custom Dense module from the previous
exercise.
d. Train this model on the GPU, and try to reach
93% accuracy on the test set. For this, you
will likely have to perform hyperparameter
search to find the right number of layers and
neurons per layer, a good learning rate and
batch size, and so on. You can optionally use
Optuna for this.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 11

1. What is the problem that Glorot initialization and He
initialization aim to fix?2. Is it OK to initialize all the weights to the same value as
long as that value is selected randomly using He
initialization?
3. Is it OK to initialize the bias terms to 0?
4. In which cases would you want to use each of the
activation functions we discussed in this chapter?
5. What may happen if you set the momentum
hyperparameter too close to 1 (e.g., 0.99999) when using
an SGD optimizer?
6. Name three ways you can produce a sparse model.
7. Does dropout slow down training? Does it slow down
inference (i.e., making predictions on new instances)?
What about MC dropout?
8. Practice training a deep neural network on the CIFAR10
image dataset:
a. Load CIFAR10 just like you loaded the
FashionMNIST dataset in Chapter 10, but using
torchvision.datasets.CIFAR10 instead of
FashionMNIST. The dataset is composed of 60,000
32 × 32–pixel color images (50,000 for training,
10,000 for testing) with 10 classes.
b. Build a DNN with 20 hidden layers of 100 neurons
each (that’s too many, but it’s the point of this
exercise). Use He initialization and the Swish
activation function (using nn.SiLU). Since this is a
classification task, you will need an output layer
with one neuron per class.
c. Using NAdam optimization and early stopping,
train the network on the CIFAR10 dataset.
Remember to search for the right learning rate
each time you change the model’s architecture or
hyperparameters.d. Now try adding batch-norm and compare the
learning curves: is it converging faster than
before? Does it produce a better model? How does
it affect training speed?
e. Try replacing batch-norm with SELU, and make
the necessary adjustments to ensure the network
self-normalizes (i.e., standardize the input
features, use LeCun normal initialization, make
sure the DNN contains only a sequence of dense
layers, etc.).
f. Try regularizing the model with alpha dropout.
Then, without retraining your model, see if you can
achieve better accuracy using MC dropout.
g. Retrain your model using 1cycle scheduling and
see if it improves training speed and model
accuracy.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 12

1. What are the advantages of a CNN over a fully connected
DNN for image classification?
2. Consider a CNN composed of three convolutional layers,
each with 3 × 3 kernels, a stride of 2, and "same" padding.The lowest layer outputs 100 feature maps, the middle one
outputs 200, and the top one outputs 400. The input
images are RGB images of 200 × 300 pixels:
a. What is the total number of parameters in the
CNN?
b. If we are using 32-bit floats, at least how much
RAM will this network require when making a
prediction for a single instance?
c. What about when training on a mini-batch of 50
images?
3. If your GPU runs out of memory while training a CNN,
what are five things you could try to solve the problem?
4. Why would you want to add a max pooling layer rather
than a convolutional layer with the same stride?
5. Can you name the main innovations in AlexNet, as
compared to LeNet-5? What about the main innovations in
GoogLeNet, ResNet, SENet, Xception, EfficientNet, and
ConvNeXt?
6. What is a fully convolutional network? How can you
convert a dense layer into a convolutional layer?
7. What is the main technical difficulty of semantic
segmentation?
8. Build your own CNN from scratch and try to achieve the
highest possible accuracy on MNIST.
9. Use transfer learning for large image classification, going
through these steps:
a. Create a training set containing at least 100 images
per class. For example, you could classify your own
pictures based on the location (beach, mountain,
city, etc.). Alternatively, you can use an existing
dataset, such as the one used in PyTorch’s transfer
learning for computer vision tutorial.b. Split it into a training set, a validation set, and a
test set.
c. Build the input pipeline, apply the appropriate
preprocessing operations, and optionally add data
augmentation.
d. Fine-tune a pretrained model on this dataset.
10. Go through PyTorch’s object detection fine-tuning tutorial.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 13

1. Can you think of a few applications for a sequence-
to-sequence RNN? What about a sequence-to-vector
RNN, and a vector-to-sequence RNN?
2. How many dimensions must the inputs of an RNN
layer have? What does each dimension represent?
What about its outputs?3. How can you build a deep sequence-to-sequence
RNN in PyTorch?
4. Suppose you have a daily univariate time series, and
you want to forecast the next seven days using an
RNN. Which architecture should you use?
5. What are the main difficulties when training RNNs?
How can you handle them?
6. Can you sketch the LSTM cell’s architecture?
7. Why would you want to use 1D convolutional layers
in an RNN?
8. Which neural network architecture could you use to
classify videos?
9. Try to tweak the Seq2SeqModel model to forecast
both rail and bus ridership for the next 14 days. The
model will now need to predict 28 values instead of
14.
10. Download the Bach chorales dataset and unzip it. It
is composed of 382 chorales composed by Johann
Sebastian Bach. Each chorale is 100 to 640 time
steps long, and each time step contains 4 integers,
where each integer corresponds to a note’s index on
a piano (except for the value 0, which means that no
note is played). Train a model—recurrent,
convolutional, or both—that can predict the next
time step (four notes), given a sequence of time
steps from a chorale. Then use this model to
generate Bach-like music, one note at a time: you
can do this by giving the model the start of a chorale
and asking it to predict the next time step, then
appending these time steps to the input sequenceand asking the model for the next note, and so on.
Also make sure to check out Google’s Coconet
model, which was used for a nice Google doodle
about Bach.
11. Train a classification model for the QuickDraw
dataset, which contains millions of sketches of
various objects. Start by downloading the simplified
data for a few classes (e.g., ant.ndjson, axe.ndjson,
and bat.ndjson). Each NDJSON file contains one
JSON object per line, which you can parse using
Python’s json.loads() function. This will give you a
list of sketches, where each sketch is represented as
a Python dictionary. In each dictionary, the
"drawing" entry contains a list of pen strokes. You
can convert this list to a 3D float tensor where the
dimensions are [strokes, x coordinates, y
coordinates]. Since an RNN takes a single sequence
as input, you will need to concatenate all the strokes
for each sketch into a single sequence. It’s best to
add an extra feature to allow the RNN to know how
far along each stroke it currently is (e.g., from 0 to
1). In other words, the model will receive a sequence
where each time step has three features: the x and y
coordinates of the pen, and the progress ratio along
the current stroke.
12. Create a dataset containing short audio recordings
of you saying “yes” or “no”, and train a binary
classification RNN on it. For example, you could:
a. Use an audio recording software such as
Audacity to record yourself saying “yes” as
many times as your patience allows, with
short pauses between each word. Create asimilar recording for the word “no”. Try to
cover the various ways you might realistically
pronounce these words in real life.
b. Load each WAV file using the
torchaudio.load() function from the
TorchAudio library. This will return a tensor
containing the audio, as well as an integer
indicating the number of samples per second.
The audio tensor has a shape of [channels,
samples]: one channel for mono, two for
stereo. Convert stereo to mono by averaging
over the channel dimension.
c. Chop each recording into individual words by
splitting at the silences. You can do this using
the torchaudio.transforms.Vad transform
(Voice Activity Detection).
d. Since the sequences are so long, it’s hard to
directly train an RNN on them, so it helps to
convert the audio to a spectrogram first. For
this, you can use the
torchaudio.transforms.MelSpectrogram
transform, which is well suited for voice. The
output is a dramatically shorter sequence,
with many more channels.
e. Now try building and training a binary
classification RNN on your yes/no dataset!
Consider sharing your dataset and model with
the world (e.g., via the Hugging Face Hub).
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 14

1. What are the pros and cons of using a stateful RNN versus a stateless RNN?
2. Why do people use encoder-decoder RNNs rather than plain sequence-to-sequence
RNNs for automatic translation?
3. How can you deal with variable-length input sequences? What about variable-
length output sequences?
4. What is beam search, and why would you use it? What tool can you use to
implement it?
5. What is an attention mechanism? How does it help?
6. When would you need to use sampled softmax?
7. Embedded Reber grammars were used by Hochreiter and Schmidhuber in their
paper about LSTMs. They are artificial grammars that produce strings such as
“BPBTSXXVPSEPE”. Check out Jenny Orr’s nice introduction to this topic, then
choose a particular embedded Reber grammar (such as the one represented on
Orr’s page), and train an RNN to identify whether a string respects that grammar
or not. You will first need to write a function capable of generating a training batch
containing about 50% strings that respect the grammar, and 50% that don’t.
8. Train an encoder-decoder model that can convert a date string from one format to
another (e.g., from “April 22, 2019” to “2019-04-22”).
Solutions to these exercises are available at the end of this chapter’s notebook, at
https://homl.info/colab-p.

## Chapter 15

1. What is the most important layer in the Transformer
architecture? What is its purpose?
2. Why does the Transformer architecture need
positional encodings?
3. What tasks are encoder-only models best at? How
about decoder-only models? And encoder-decoder
models?
4. What is the most important technique used to
pretrain BERT?
5. Can you name four BERT variants and explain their
main benefits?
6. What is the main task used to pretrain GPT and its
successors?
7. The generate() method has many arguments,
including do_sample, top_k, top_p, temperature,
and num_beams. What do these five arguments do?
8. What is prompt engineering? Can you describe five
prompt engineering techniques?
9. What are the main steps to build a chatbot, starting
from a pretrained decoder-only model?10. How can a chatbot use tools like a calculator or web
search?
11. What is MCP used for?
12. Fine-tune BERT for sentiment analysis on the IMDb
dataset.
13. Fine-tune GPT-2 on the Shakespeare dataset (from
Chapter 14), then generate Shakespeare-like text.
14. Download the Wikipedia Movie Plots dataset, and
use SBERT to embed every movie description. Then
write a function that takes a search query, embeds
it, finds the most similar embeddings, and lists the
corresponding movies.
15. Use an instruction-tuned model such as Qwen-7B-
Instruct to build a little chatbot which acts like a
movie expert. Then try adding some RAG
functionality, for example by automatically injecting
the most relevant movie plot into the prompt (see
the previous exercise).
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 16

1. Can you describe the original ViT’s architecture?
Why does it matter?
2. What tasks are regular ViTs (meaning
nonhierarchical) best used for? What are their
limitations?
3. What is the main innovation in DeiT? Is this idea
generalizable to other architectures?
4. What are some examples of hierarchical ViTs? What
kind of tasks are they good for?
5. How do PVTs and Swin Transformers reduce the
computational cost of processing high-resolution
images?
6. How does DINO work? What changed in DINOv2?
When would you want to use DINOv2?
7. What is the objective of the JEPA architecture? How
does it work?
8. What is a multimodal model? Can you give five
examples of multimodal tasks?
9. Explain what the fusion and alignment problems are
in multimodal learning. Why are transformers well
suited to tackle them?10. Can you write a one-line summary of the main ideas
in VideoBERT, ViLBERT, CLIP, DALL·E, Perceiver
IO, Flamingo, and BLIP-2?
11. If you are using a Perceiver IO model and you
double the length of the inputs and the outputs,
approximately how much more computation will be
required?
12. Try fine-tuning a pretrained ViT model on the Food
101 dataset (torchvision.datasets.Food101).
What accuracy can you reach? How about using a
CLIP model, zero-shot?
13. Create a simple search engine for your own photos:
first, write a function that uses a CLIP model to
embed all of your photos and saves the resulting
vectors. Next, write a function that takes a search
query (text or image), embeds it using CLIP, then
finds the most similar photo embeddings and
displays the corresponding photos. You can
manually implement the similarity search algorithm,
or a dedicated library such as the FAISS library or
even a full-blown vector database.
14. Use BLIP-2 to automatically caption all of your
photos.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 17

Skip - author's intsruction

## Chapter 18

1. What are the main tasks that autoencoders are used for?
2. Suppose you want to train a classifier, and you have
plenty of unlabeled training data but only a few
thousand labeled instances. How can autoencoders
help? How would you proceed?3. If an autoencoder perfectly reconstructs the inputs, is it
necessarily a good autoencoder? How can you evaluate
the performance of an autoencoder?
1. What are undercomplete and overcomplete
autoencoders? What is the main risk of an excessively
undercomplete autoencoder? What about the main risk
of an overcomplete autoencoder?
1. How do you tie weights in a stacked autoencoder? What
is the point of doing so?
1. What is a generative model? Can you name a type of
generative autoencoder?
1. What is a GAN? Can you name a few tasks where GANs
can shine?
1. What are the main difficulties when training GANs?
2. What are diffusion models good at? What is their main
limitation?
1.  Try using a denoising autoencoder to pretrain an image
classifier. You can use MNIST (the simplest option), or a
more complex image dataset such as CIFAR10 if you
want a bigger challenge. Regardless of the dataset
you’re using, follow these steps:
a. Split the dataset into a training set and a test set.
Train a deep denoising autoencoder on the full
training set.
b. Check that the images are fairly well
reconstructed. Visualize the images that most
activate each neuron in the coding layer.
c. Build a classification DNN, reusing the lower
layers of the autoencoder. Train it using only 500
images from the training set. Does it perform
better with or without pretraining?11. Train a variational autoencoder on the image dataset of
your choice, and use it to generate images. Alternatively,
you can try to find an unlabeled dataset that you are
interested in and see if you can generate new samples.
12. Train a DCGAN to tackle the image dataset of your
choice, and use it to generate images. Add experience
replay and see if this helps.
13.  Train a diffusion model on your preferred image dataset
(e.g., torchvision.datasets.Flowers102), and
generate nice images. Next, add the image class as an
extra input to the model, and retrain it: you should now
be able to control the class of the generated image.
Solutions to these exercises are available at the end of this
chapter’s notebook, at https://homl.info/colab-p.

## Chapter 19

1. How would you define reinforcement learning? How is it different
from regular supervised or unsupervised learning?
2. Can you think of three possible applications of RL that were not
mentioned in this chapter? For each of them, what is the
environment? What is the agent? What are some possible actions?
What are the rewards?
3. What is the discount factor? Can the optimal policy change if you
modify the discount factor?
4. How do you measure the performance of a reinforcement learning
agent?
5. What is the credit assignment problem? When does it occur? How
can you alleviate it?
6. What is the point of using a replay buffer?
7. What is an off-policy RL algorithm? What are the benefits?
8. What is a model-based RL algorithm? Can you give some
examples?9. Use policy gradients to solve Gymnasium’s LunarLander-v2
environment.
10. Solve the BipedalWalker-v3 environment using the RL algorithm
of your choice.
11. If you have about $100 to spare, you can purchase a Raspberry Pi
3 plus some cheap robotics components, install PyTorch on the Pi,
and go wild! Start with simple goals, like making the robot turn
around to find the brightest angle (if it has a light sensor) or the
closest object (if it has a sonar sensor), and move in that
direction. Then you can start using deep learning. For example, if
the robot has a camera, you can try to implement an object
detection algorithm so it detects people and moves toward them.
You can also try to use RL to make the agent learn on its own how
to use the motors to achieve that goal. Have fun!
Solutions to these exercises are available at the end of this chapter’s
notebook, at https://homl.info/colab-p.
