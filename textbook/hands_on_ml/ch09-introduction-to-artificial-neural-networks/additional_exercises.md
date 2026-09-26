# Chapter 9 — Introduction to Artificial Neural Networks

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. What is the perceptron learning rule, and what is Scikit-Learn's Perceptron class equivalent to?

A perceptron is a single layer of threshold logic units (TLUs): each one computes z = wᵀx + b and outputs step(z). It's trained one instance at a time, and each output neuron that made a wrong prediction gets its weights adjusted:

wᵢ,ⱼ ← wᵢ,ⱼ + η(yⱼ − ŷⱼ)xᵢ

Here wᵢ,ⱼ connects input i to neuron j, xᵢ is the instance's ith input value, yⱼ and ŷⱼ are neuron j's target and predicted outputs, and η is the learning rate. A correct prediction leaves the weights unchanged; a wrong one strengthens the connections from inputs that would have pushed toward the right answer, an error-driven twist on Hebb's rule. If the classes are linearly separable, training is guaranteed to converge.

This is essentially stochastic gradient descent: Scikit-Learn's Perceptron is equivalent to SGDClassifier(loss="perceptron", learning_rate="constant", eta0=1, penalty=None), so there's no regularization by default.

## 2. Why does an MLP need nonlinear activation functions in its hidden layers?

Stacking linear layers without anything nonlinear in between just gives another linear layer: (XW₁ + b₁)W₂ + b₂ = X(W₁W₂) + (b₁W₂ + b₂), which is a single layer with weights W₁W₂ and bias b₁W₂ + b₂. So however deep it is, such a network can only learn linear functions and linear decision boundaries; it can't even solve XOR.

Putting a nonlinear activation such as ReLU after each hidden layer prevents this collapse, so each layer can build on the previous one. With nonlinear activations, a large enough network can approximate any continuous function.

## 3. Why must an MLP's hidden-layer weights be initialized randomly rather than all to zero?

If all the weights and biases in a layer start out equal (for example, all zero), every neuron in that layer computes the same output and receives exactly the same gradient during backpropagation, so every update keeps them identical. However many neurons the layer has, it behaves like a single neuron.

Random initialization breaks this symmetry, letting each neuron learn a different feature. The biases can start at zero as long as the weights are random. This only matters once there are hidden layers: a plain linear or logistic regression model trains fine from all-zero weights.

## 4. Why does tanh often train faster than the sigmoid, and why did ReLU become the default hidden-layer activation despite its flaws?

The sigmoid σ(z) = 1 / (1 + exp(−z)) outputs values between 0 and 1. tanh(z) = 2σ(2z) − 1 has the same S-shape but ranges from −1 to 1, so each layer's outputs tend to be roughly centered around 0 at the start of training, which often speeds up convergence.

ReLU(z) = max(0, z) has flaws: it isn't differentiable at z = 0 (the abrupt change of slope can make gradient descent bounce around), and its gradient is 0 for negative inputs. But it's very fast to compute and works very well in practice, and since its output has no maximum value, it avoids some of the gradient problems of S-shaped functions, which flatten out when their inputs get large in magnitude. It's the default for most architectures, Transformers being a notable exception.

## 5. How do you choose the output layer and the loss function for a regression MLP?

- Output neurons: one per value to predict, e.g., two for an object's center coordinates, or four to add its bounding box's width and height.
- Output activation: usually none, so the model can output any value. Use ReLU or softplus(z) = log(1 + exp(z)), a smooth variant of ReLU, to guarantee positive outputs. Use sigmoid or tanh to keep predictions within a range, after scaling the targets to 0–1 or −1–1, respectively.
- Loss: usually the MSE. If the training set has many outliers, prefer the Huber loss: quadratic for errors below a threshold δ (typically 1), linear above it. The linear part makes it less sensitive to outliers than the MSE, and the quadratic part makes it converge faster and more precisely than the mean absolute error.

## 6. Sketch training a regression MLP with Scikit-Learn's MLPRegressor, and explain its key hyperparameters.

A pipeline standardizes the inputs, which matters because gradient descent converges poorly when features have very different scales:

```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50],
                       early_stopping=True, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

- hidden_layer_sizes lists the number of neurons in each hidden layer; the input and output sizes adapt to the data when training starts. Hidden layers use ReLU, and the output layer has no activation.
- early_stopping=True holds out 10% of the training set (validation_fraction) and stops once the validation score hasn't improved for 10 consecutive epochs (n_iter_no_change); best_validation_score_ keeps the best score.
- It minimizes the MSE using the Adam optimizer, plus a small ℓ2 penalty whose strength is alpha (0.0001 by default).
- score() returns the R² score, as for every Scikit-Learn regressor.

## 7. How does MLPClassifier differ from MLPRegressor, and why scale Fashion MNIST pixels with a MinMaxScaler rather than a StandardScaler?

MLPClassifier has the same API and hyperparameters, but for multiclass tasks its output layer uses softmax, it minimizes the cross-entropy rather than the MSE, score() returns the accuracy rather than R², and predict_proba() returns the estimated class probabilities.

The MinMaxScaler maps the pixel intensities from 0–255 to 0–1, a range that suits MLPClassifier's defaults, such as its learning rate and weight initialization scale. A StandardScaler would rescale every pixel to unit variance, including pixels that barely vary across images (like those near the edges, which are almost always white). That would inflate them and give them more importance than they deserve. MinMaxScaler often works better for images, but it's worth checking on your data.

## 8. Why should you be wary of a neural net's estimated class probabilities, and how can label smoothing help?

Neural nets tend to be overconfident, especially if they're trained a bit too long. A classifier that's about 90% accurate can still assign close to 100% probability to its wrong predictions, so don't take predict_proba() outputs at face value.

Label smoothing replaces the one-hot targets with softer ones: the target class gets slightly less than 1 (e.g., 0.9), and the remainder is spread evenly across the other classes (0.1/9 each, with 10 classes). The model is no longer rewarded for pushing its probabilities all the way to 0 or 1, which reduces overconfidence.

## 9. Why are deep networks usually more parameter-efficient than shallow ones, and how does that make transfer learning possible?

In theory one hidden layer can model very complex functions if it has enough neurons, but deep networks can model them with exponentially fewer neurons because each layer reuses and combines the features of the layer below. In a face classifier, the lowest layers might detect lines and arcs, the next ones shapes such as squares and circles, the next ones eyes and noses, and the top layer uses these to classify faces. This hierarchy helps deep nets converge faster and generalize better.

Since the lower layers learn generic low-level features, a network for a related task (say, recognizing hairstyles) can start with its lower layers initialized to a trained face network's weights, and only has to learn the higher-level structure. That's transfer learning: training is much faster and needs far less data.

## 10. How should you size an MLP's hidden layers? Explain the "stretch pants" approach and why bottleneck layers are risky.

The input and output sizes are dictated by the task. For hidden layers, the old pyramid shape (fewer neurons in each successive layer) has largely been abandoned: giving all hidden layers the same size usually works as well or better, and leaves just one hyperparameter to tune (a slightly larger first layer sometimes helps).

Stretch pants approach: rather than hunting for the exact right size, build a model with somewhat more layers and neurons than you need, and rely on early stopping and other regularization to keep it from overfitting. This avoids bottlenecks: a layer that's too narrow can't preserve all the useful information from the inputs, and the layers above can never recover what it lost. For example, make the first hidden layer larger than the number of PCA dimensions needed to keep 95% of the variance. Adding layers generally pays off more than adding neurons per layer.

## 11. Walk through finding a good learning rate by exponentially increasing it during a short training run.

1. Start training with a tiny learning rate, such as 10⁻⁵.
2. After each iteration, multiply the learning rate by a constant factor so it reaches a huge value, such as 10, after a few hundred iterations. To go from 10⁻⁵ to 10 in 500 iterations, the factor is (10 / 10⁻⁵) to the power 1/500, about 1.028.
3. Plot the loss against the learning rate, with a log scale for the learning rate. The loss drops at first, then shoots back up once the learning rate gets too large.
4. Pick a learning rate roughly 10 times lower than the turning point where the loss starts climbing.
5. Reinitialize the model and train it normally with that learning rate.

As a rule of thumb, the optimal learning rate is about half the maximum learning rate, above which training diverges.

## 12. What are the trade-offs between large and small batch sizes, and what's a sensible strategy for choosing one?

Large batches let hardware accelerators like GPUs process many instances in parallel, so training sees more instances per second; hence the common advice to use the largest batch size that fits in GPU memory. But large batches can make training unstable, especially early on and with smaller models, and the resulting model may generalize worse.

The research is mixed: some found that small batches (2 to 32) produced better models in less time, while others trained with very large batches (up to 8,192) without a generalization gap, using tricks such as learning rate warmup (starting with a small learning rate and ramping it up).

A sensible strategy: try a large batch size with warmup, and switch to a smaller one if training is unstable or the final performance is disappointing. The best learning rate depends on the batch size, so re-tune it whenever you change the batch size.
