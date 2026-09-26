# Chapter 9 — Introduction to Artificial Neural Networks

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. Answers below
are the author's, from the exercise solutions at the end of
`handson-mlp-main/09_artificial_neural_networks.ipynb`, converted to plain
text for the flashcard app.

Text in [brackets] is added, not the author's: it describes a figure the
flashcard can't show.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text. Solutions are in the same
notebook unless noted.

- 1. Explore the TensorFlow Playground: what the layers learn, tanh vs.
  ReLU, local minima, too-small vs. big-enough networks, and vanishing
  gradients on the spiral dataset. No code: the author's solution is just to
  go explore.
- 10. Train a deep MLP on the CoverType dataset
  (sklearn.datasets.fetch_covtype()) and aim for over 93% test accuracy,
  tuning by hand and/or with RandomizedSearchCV.

## 2. Draw an ANN using the original artificial neurons (like the ones in Figure 9-3) that computes A ⊕ B (where ⊕ represents the XOR operation). Hint: A ⊕ B = (A ∧ ¬ B) ∨ (¬ A ∧ B).

Here is a neural network based on the original artificial neurons that computes A ⊕ B (where ⊕ represents the exclusive OR), using the fact that A ⊕ B = (A ∧ ¬ B) ∨ (¬ A ∧ B):

[The author's diagram, in words: two hidden neurons feed one output neuron. One hidden neuron computes A ∧ ¬ B (two connections from A, plus an inhibitory connection from B); the other computes ¬ A ∧ B (the same with A and B swapped). The output neuron ORs them: it has two connections from each hidden neuron, so either one alone activates it. As in Figure 9-3, a neuron fires when at least two of its input connections are active, unless an inhibitory connection is active.]

There are other solutions—for example, using the fact that A ⊕ B = (A ∨ B) ∧ ¬(A ∧ B), or the fact that A ⊕ B = (A ∨ B) ∧ (¬ A ∨ ¬ B), and so on.

## 3. Why is it generally preferable to use a logistic regression classifier rather than a classic perceptron (i.e., a single layer of threshold logic units trained using the perceptron training algorithm)? How can you tweak a perceptron to make it equivalent to a logistic regression classifier?

A classical Perceptron will converge only if the dataset is linearly separable, and it won't be able to estimate class probabilities. In contrast, a Logistic Regression classifier will generally converge to a reasonably good solution even if the dataset is not linearly separable, and it will output class probabilities. If you change the Perceptron's activation function to the sigmoid activation function (or the softmax activation function if there are multiple neurons), and if you train it using Gradient Descent (or some other optimization algorithm minimizing the cost function, typically cross entropy), then it becomes equivalent to a Logistic Regression classifier.

## 4. Why was the sigmoid activation function a key ingredient in training the first MLPs?

The sigmoid activation function was a key ingredient in training the first MLPs because its derivative is always nonzero, so Gradient Descent can always roll down the slope. When the activation function is a step function, Gradient Descent cannot move, as there is no slope at all.

## 5. Name three popular activation functions. Can you draw them?

Popular activation functions include the step function, the sigmoid function, the hyperbolic tangent (tanh) function, and the Rectified Linear Unit (ReLU) function (see Figure 9-8). See Chapter 11 for other examples, such as ELU and variants of the ReLU function.

[What Figure 9-8 shows: the step function is flat at 0, then jumps to 1 at z = 0; the sigmoid σ(z) = 1 / (1 + exp(–z)) is a smooth S-curve from 0 to 1; tanh(z) = 2σ(2z) – 1 has the same S-shape but goes from –1 to 1; ReLU(z) = max(0, z) is 0 for negative z and a straight line of slope 1 for positive z.]

## 6. Suppose you have an MLP composed of one input layer with 10 passthrough neurons, followed by one hidden layer with 50 artificial neurons, and finally one output layer with 3 artificial neurons. All artificial neurons use the ReLU activation function. a. What is the shape of the input matrix X? b. What are the shapes of the hidden layer’s weight matrix Wh and bias vector bh? c. What are the shapes of the output layer’s weight matrix Wo and bias vector bo? d. What is the shape of the network’s output matrix Y? e. Write the equation that computes the network’s output matrix Y as a function of X, Wh, bh, Wo, and bo.

Considering the MLP described in the question, composed of one input layer with 10 passthrough neurons, followed by one hidden layer with 50 artificial neurons, and finally one output layer with 3 artificial neurons, where all artificial neurons use the ReLU activation function:
- The shape of the input matrix X is m × 10, where m represents the training batch size.
- The shape of the hidden layer's weight matrix Wₕ is 10 × 50, and the length of its bias vector bₕ is 50.
- The shape of the output layer's weight matrix Wₒ is 50 × 3, and the length of its bias vector bₒ is 3.
- The shape of the network's output matrix Y is m × 3.
- Y = ReLU(ReLU(X Wₕ + bₕ) Wₒ + bₒ). Recall that the ReLU function just sets every negative number in the matrix to zero. Also note that when you are adding a bias vector to a matrix, it is added to every single row in the matrix, which is called broadcasting.

## 7. How many neurons do you need in the output layer if you want to classify email into spam or ham? What activation function should you use in the output layer? If instead you want to tackle MNIST, how many neurons do you need in the output layer, and which activation function should you use? What about for getting your network to predict housing prices, as in Chapter 2?

To classify email into spam or ham, you just need one neuron in the output layer of a neural network—for example, indicating the probability that the email is spam. You would typically use the sigmoid activation function in the output layer when estimating a probability. If instead you want to tackle MNIST, you need 10 neurons in the output layer, and you must replace the sigmoid function with the softmax activation function, which can handle multiple classes, outputting one probability per class. If you want your neural network to predict housing prices like in Chapter 2, then you need one output neuron, using no activation function at all in the output layer. Note: when the values to predict can vary by many orders of magnitude, you may want to predict the logarithm of the target value rather than the target value directly. Simply computing the exponential of the neural network's output will give you the estimated value (since exp(log v) = v).

## 8. What is backpropagation and how does it work? What is the difference between backpropagation and reverse-mode autodiff?

Backpropagation is a technique used to train artificial neural networks. It first computes the gradients of the cost function with regard to every model parameter (all the weights and biases), then it performs a Gradient Descent step using these gradients. This backpropagation step is typically performed thousands or millions of times, using many training batches, until the model parameters converge to values that (hopefully) minimize the cost function. To compute the gradients, backpropagation uses reverse-mode autodiff (although it wasn't called that when backpropagation was invented, and it has been reinvented several times). Reverse-mode autodiff performs a forward pass through a computation graph, computing every node's value for the current training batch, and then it performs a reverse pass, computing all the gradients at once (see Appendix A for more details). So what's the difference? Well, backpropagation refers to the whole process of training an artificial neural network using multiple backpropagation steps, each of which computes gradients and uses them to perform a Gradient Descent step. In contrast, reverse-mode autodiff is just a technique to compute gradients efficiently, and it happens to be used by backpropagation.

## 9. Can you list all the hyperparameters you can tweak in a basic MLP? If the MLP overfits the training data, how could you tweak these hyperparameters to try to solve the problem?

Here is a list of all the hyperparameters you can tweak in a basic MLP: the number of hidden layers, the number of neurons in each hidden layer, and the activation function used in each hidden layer and in the output layer. In general, the ReLU activation function (or one of its variants; see Chapter 11) is a good default for the hidden layers. For the output layer, in general you will want the sigmoid activation function for binary classification, the softmax activation function for multiclass classification, or no activation function for regression. If the MLP overfits the training data, you can try reducing the number of hidden layers and reducing the number of neurons per hidden layer.
