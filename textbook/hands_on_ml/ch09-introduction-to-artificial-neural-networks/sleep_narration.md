# Chapter 9 — Introduction to Artificial Neural Networks, to fall asleep to

Chapter nine. Introduction to artificial neural networks.

---

Birds inspired flight, and burdock plants inspired Velcro. Artificial neural networks, or ANNs, are machine learning models inspired by the networks of biological neurons in the brain. Planes were inspired by birds but do not flap their wings, and in the same way ANNs have become quite different from their biological cousins. Some researchers argue for dropping the biological analogy altogether, for example by saying units instead of neurons, so as not to restrict creativity to biologically plausible systems.

ANNs are at the core of deep learning. They are versatile, powerful, and scalable, which makes them ideal for large, highly complex tasks: classifying billions of images, speech recognition, chatbots such as ChatGPT or Claude, recommending videos to hundreds of millions of users a day, as YouTube does, or learning how proteins fold, as DeepMind's AlphaFold does.

The chapter goes from the first ANN architectures to multilayer perceptrons, or MLPs, which are heavily used today, built here with Scikit-Learn. The next chapter switches to PyTorch, a much more flexible and efficient library for neural networks.

---

A short history of neural networks. ANNs were first introduced in 1943 by the neurophysiologist Warren McCulloch and the mathematician Walter Pitts. Their paper presented a simplified computational model of how biological neurons might work together in animal brains to perform complex computations using propositional logic. It was the first artificial neural network architecture.

Early successes led to a widespread belief that truly intelligent machines were near. When it became clear in the 1960s that this promise would go unfulfilled for a while, funding went elsewhere, and ANNs entered a long winter. In the early 1980s, new architectures and better training techniques revived interest in connectionism, the study of neural networks. Progress was slow, though, and by the 1990s other powerful techniques, such as support vector machines, seemed to offer better results and stronger theoretical foundations, so neural network research was put on hold again.

The current wave is likely different, for several reasons. There is now a huge quantity of training data, and ANNs frequently outperform other techniques on very large and complex problems. Computing power has grown tremendously since the 1990s, partly through Moore's law, under which the number of components in integrated circuits has doubled about every two years over the last fifty years. The gaming industry also drove the production of powerful graphics processing units, or GPUs, by the millions. GPUs were designed for graphics, but neural networks perform similar computations, such as large matrix multiplications, so GPUs accelerate them too. Cloud platforms have made this power accessible to everyone.

Training algorithms have improved; they differ only slightly from those of the 1990s, but the small tweaks have had a huge impact. Some theoretical limitations turned out to be benign in practice. Training was expected to get stuck in local optima, but especially for larger networks, local optima often perform almost as well as the global optimum.

The Transformer architecture, invented in 2017, has been a game changer. Unlike earlier, more specialized architectures, it can process and generate text, images, audio, and more, and it performs well on tasks from robotics to protein folding. It scales well, which made it possible to train very large foundation models that can be reused across many tasks, either with a bit of fine-tuning, which is transfer learning, or by prompting them the right way, which is in-context learning. Prompts can include a few examples of the task, which is few-shot learning, or ask the model to reason step by step, which is chain-of-thought prompting.

Finally, ANNs have entered a virtuous circle: impressive products attract attention and funding, which produce more progress and more products. Since chatbots were released, the public uses AI assistants daily, and big tech companies compete fiercely for this market.

---

Biological neurons. A biological neuron is an unusual-looking cell found mostly in animal brains. It has a cell body containing the nucleus and most of the cell's complex components, many branching extensions called dendrites, and one very long extension called the axon. The axon may be just a few times longer than the cell body, or up to tens of thousands of times longer. Near its end, the axon splits into many branches called telodendria, and at their tips are tiny structures called synaptic terminals, or synapses, which connect to the dendrites or cell bodies of other neurons.

Biological neurons produce short electrical impulses called action potentials, or simply signals. These travel along the axons and make the synapses release chemical signals called neurotransmitters. When a neuron receives enough neurotransmitters within a few milliseconds, it fires its own impulses. Some neurotransmitters inhibit firing instead.

Each neuron behaves fairly simply, but neurons form a vast network of billions, each typically connected to thousands of others, and together they perform highly complex computations, much as a complex anthill emerges from simple ants. Parts of the brain have been mapped, and neurons are often organized in consecutive layers, especially in the cerebral cortex, the brain's outer layer.

---

Logical computations with neurons. McCulloch and Pitts proposed a very simple model of the biological neuron, later known as an artificial neuron. It has one or more binary inputs, each on or off, and one binary output. It activates its output when more than a certain number of its inputs are active. McCulloch and Pitts showed that a network of such neurons can compute any logical proposition.

The book's examples assume a neuron activates when at least two of its input connections are active. If neuron A has two connections to neuron C, C is active exactly when A is: the identity function. If A and B each have one connection to C, C activates only when both are active: a logical AND. If A and B each have two connections to C, C activates when either is active: a logical OR. If an input connection can inhibit a neuron, as with biological neurons, a network can activate C only when A is active and B is off. With A always active, this is a logical NOT of B. Such networks can be combined to compute complex logical expressions.

---

The perceptron. The perceptron, invented in 1957 by Frank Rosenblatt, is one of the simplest ANN architectures. It is based on a slightly different artificial neuron called a threshold logic unit, or TLU, sometimes called a linear threshold unit. Its inputs and output are numbers instead of on or off values, and each input connection has a weight.

A TLU first computes a linear function of its inputs: each input times its weight, all added together, plus a bias term. It then applies a step function to the result. This is almost like logistic regression, except that it uses a step function instead of the logistic function. As in logistic regression, the model parameters are the input weights and the bias term.

The most common step function in perceptrons is the Heaviside step function, which outputs zero when its input is negative and one when its input is zero or positive. Sometimes the sign function is used instead, which outputs minus one for negative inputs, zero for zero, and plus one for positive inputs.

A single TLU can perform simple linear binary classification: if the linear function of its inputs exceeds a threshold, it outputs the positive class, otherwise the negative class, much like logistic regression or a linear support vector machine. A single TLU could classify iris flowers from petal length and width; training it means finding the right two weights and bias.

A perceptron is one or more TLUs in a single layer, with every TLU connected to every input. Such a layer is called a fully connected layer, or a dense layer. The inputs form the input layer, and the layer of TLUs, which produces the final outputs, is the output layer. A perceptron with two inputs and three outputs can classify instances into three binary classes at once, which makes it a multilabel classifier. It can also be used for multiclass classification.

A fully connected layer's outputs can be computed for many instances at once. The input matrix, with one row per instance and one column per feature, is multiplied by the weight matrix, with one row per input feature and one column per neuron; in PyTorch it is transposed. The bias vector, one bias per neuron, is then added, and the activation function, a step function for TLUs, is applied to each item. The result has one row per instance and one column per neuron. Adding a vector to a matrix is undefined in mathematics, but data science allows broadcasting, which adds the vector to every row.

---

Training the perceptron. Rosenblatt's training algorithm was largely inspired by Hebb's rule. In his 1949 book The Organization of Behavior, Donald Hebb suggested that when a biological neuron often triggers another neuron, the connection between them grows stronger. Siegrid Löwel later summarized the idea in the phrase "cells that fire together, wire together": the connection weight between two neurons tends to increase when they fire simultaneously. This became known as Hebb's rule, or Hebbian learning.

Perceptrons use a variant of this rule that takes the network's error into account. The perceptron is fed one training instance at a time, and for every output neuron that predicted wrongly, it reinforces the connection weights from the inputs that would have contributed to the correct prediction. Each weight between an input and an output neuron is updated by adding the learning rate times the neuron's error times the input value, where the error is the target output minus the neuron's actual output.

Each output neuron's decision boundary is linear, so perceptrons cannot learn complex patterns, just like logistic regression classifiers. But Rosenblatt showed that if the training instances are linearly separable, the algorithm converges to a solution. This is called the perceptron convergence theorem. The solution is not unique, since infinitely many hyperplanes can separate linearly separable data.

Scikit-Learn provides a perceptron class that works as expected; the book trains one on the iris dataset to detect Iris setosa from petal length and width. The perceptron learning algorithm strongly resembles stochastic gradient descent. In fact, Scikit-Learn's perceptron class is equivalent to its stochastic gradient descent classifier with the perceptron loss, a constant learning rate of one, and no regularization.

Unlike logistic regression classifiers, perceptrons do not output a class probability, which is one reason to prefer logistic regression. Perceptrons also use no regularization by default, and training stops as soon as there are no prediction errors on the training set, so they typically generalize worse than logistic regression or a linear support vector machine. They may train a bit faster, though.

---

The limits of the perceptron. In their 1969 monograph Perceptrons, Marvin Minsky and Seymour Papert highlighted serious weaknesses of perceptrons, in particular their inability to solve some trivial problems, such as the exclusive OR, or XOR, classification problem. This is true of any linear classification model, such as logistic regression, but researchers had expected much more from perceptrons. Some were so disappointed that they dropped neural networks altogether in favor of more formal approaches such as logic, problem solving, and search. The lack of practical applications did not help.

Some limitations of perceptrons can be eliminated by stacking multiple perceptrons. The resulting ANN is called a multilayer perceptron, or MLP.

---

The multilayer perceptron. An MLP can solve the XOR problem. For inputs of zero and zero, or one and one, the network outputs zero, and for inputs of zero and one, or one and zero, it outputs one.

An MLP is made of one input layer, one or more layers of artificial neurons called hidden layers, originally TLUs, and a final layer of artificial neurons called the output layer. Layers close to the input are called the lower layers, and layers close to the outputs are called the upper layers. The signal flows only one way, from the inputs to the outputs, so this architecture is an example of a feedforward neural network, or FNN.

When an ANN contains a deep stack of hidden layers, it is called a deep neural network, or DNN. In the 1990s, more than two hidden layers counted as deep. Today ANNs commonly have dozens or hundreds of layers, so the definition is fuzzy. The field of deep learning studies DNNs, and more generally models containing deep stacks of computations, though many people talk about deep learning whenever neural networks are involved, even shallow ones.

---

Backpropagation. For many years researchers tried and failed to train MLPs. In the early 1960s, several discussed using gradient descent, but that requires the gradients of the model's error with regard to the model parameters, and it was not clear how to compute them efficiently for such a complex model with so many parameters, especially on the computers of the time.

In 1970, Seppo Linnainmaa introduced in his master's thesis a technique to compute all the gradients automatically and efficiently. It is now called reverse-mode automatic differentiation, or reverse-mode autodiff. In just two passes through the network, one forward and one backward, it computes the gradient of the network's error with regard to every model parameter. In other words, it finds how each connection weight and each bias should be tweaked to reduce the error. These gradients are then used for a gradient descent step. Repeating this, computing gradients and taking a step, gradually lowers the error until it reaches a minimum. This combination of reverse-mode autodiff and gradient descent is called backpropagation, or backprop.

Learning to shoot a basketball is an analogy. Throwing the ball is the forward pass, and seeing it go far to the right is the error computation. Working out that the arm, then the upper body, then the feet should turn a little is the backward pass, going down through the layers. Actually moving is the gradient descent step. Smaller errors mean smaller adjustments, and with practice the ball goes in every time.

Reverse-mode autodiff suits functions with many variables, such as weights and biases, and few outputs, such as one loss. Backpropagation applies to all sorts of computational graphs; Linnainmaa's thesis was not about neural networks at all. In 1985, David Rumelhart, Geoffrey Hinton, and Ronald Williams published a paper analyzing how backpropagation allows neural networks to learn useful internal representations. The results were so impressive that backpropagation quickly became popular, and over forty years later it is still by far the most popular training technique for neural networks.

---

How backpropagation works, in more detail. It handles one mini-batch at a time and goes through the full training set multiple times. Each pass through the training set is called an epoch. A mini-batch of thirty-two instances with one hundred features each is a matrix with thirty-two rows and one hundred columns.

For each mini-batch, the algorithm computes the output of every neuron in the first hidden layer. With fifty neurons, that output is a matrix with one row per instance and fifty columns. It is passed to the next layer, and so on up to the output layer. This is the forward pass: exactly like making predictions, except that all intermediate results are kept for the backward pass.

Next, a loss function compares the desired output with the network's actual output and returns a measure of the error.

The algorithm then computes how much each output layer parameter contributed to the error, analytically, with the chain rule of calculus, which makes this step fast and precise. It then measures how much of these contributions came from each connection in the layer below, again with the chain rule, and keeps working backward to the input layer. This reverse pass propagates the error gradient backward through the network, which gives the algorithm its name, and yields one gradient per weight and bias.

Finally, a gradient descent step tweaks all the weights and biases using those gradients.

The hidden layers' weights must be initialized randomly, or training fails. If all weights and biases start at zero, all neurons in a layer are identical, receive identical updates, and stay identical, so each layer behaves like a single neuron. Random initialization breaks the symmetry, so backpropagation can train a diverse team of neurons.

---

Activation functions. For backprop to work, Rumelhart and his colleagues made a key change to the MLP: they replaced the step function with the logistic function, also called the sigmoid function, which equals one divided by one plus the exponential of minus z. This was essential because the step function has only flat segments, so there is no gradient to work with, and gradient descent cannot move on a flat surface. The sigmoid has a well-defined nonzero derivative everywhere, so gradient descent can make progress at every step.

Backpropagation works well with many other activation functions. The hyperbolic tangent function, tanh, equals two times the sigmoid of two z, minus one. Like the sigmoid, it is S-shaped, continuous, and differentiable, but its output ranges from minus one to one instead of zero to one. That range tends to keep each layer's output roughly centered around zero at the start of training, which often speeds up convergence.

The rectified linear unit function, ReLU, outputs the larger of zero and z. It is continuous but not differentiable at zero, where the slope changes abruptly, which can make gradient descent bounce around, and its derivative is zero for negative inputs. In practice it works very well and is fast to compute, so it is the default for most architectures, except the Transformer. Having no maximum output value also helps reduce some issues during gradient descent. Biological neurons seem to use a roughly S-shaped activation, so researchers long stuck with sigmoids, but ReLU generally works better: the biological analogy was perhaps misleading.

Activation functions are needed because chaining linear transformations only gives another linear transformation. If f of x is two x plus three, and g of x is five x minus one, then f applied to g of x is ten x plus one, still linear. Without nonlinearity between layers, even a deep stack of layers is equivalent to a single layer and cannot solve very complex problems. A large enough DNN with nonlinear activations can, in theory, approximate any continuous function.

---

Regression MLPs. MLPs can handle many tasks, but the most common are regression and classification. To predict a single value, such as the price of a house from its features, an MLP needs a single output neuron, whose output is the predicted value. Multivariate regression, predicting several values at once, needs one output neuron per output dimension. Locating the center of an object in an image needs two coordinates, so two output neurons. Adding a bounding box needs the object's width and height too, so four output neurons.

The book uses Scikit-Learn's MLP regressor class to build an MLP with three hidden layers of fifty neurons each, trained on a simpler version of the California housing dataset: only numerical features, no missing values, and targets in units of one hundred thousand dollars. The input and output sizes adjust automatically to the data when training starts. The model uses ReLU in all hidden layers and no activation function on the output layer.

Neural networks have many parameters and tend to overfit. With early stopping enabled, the MLP regressor sets aside ten percent of the training data and evaluates the model on it at each epoch, stopping when the validation score has not improved for ten epochs. Both numbers are adjustable.

A pipeline standardizes the input features first, because gradient descent converges poorly when features have very different scales. The MLP regressor minimizes the mean squared error with a variant of gradient descent called Adam, plus a tiny amount of L2 regularization controlled by the alpha hyperparameter.

Training takes forty-five epochs. The training loss, roughly half the mean squared error plus the regularization term, goes down at each epoch, and the validation score generally goes up. Like every Scikit-Learn regressor, it is scored with R squared, the ratio of variance explained by the model, which reaches close to eighty percent on the validation set, fairly good for this task. The test root mean squared error, or RMSE, is about zero point five three, comparable to a random forest, with quite a few errors, especially on larger targets.

---

Output activations and losses for regression. Without an output activation function, the MLP can output any value, which is generally fine. To guarantee positive outputs, the output layer can use ReLU, or softplus, a smooth variant of ReLU equal to the log of one plus the exponential of z. Softplus is close to zero when z is negative, and close to z when z is positive. To guarantee predictions within a given range, use the sigmoid or the hyperbolic tangent, and scale the targets to match: zero to one for the sigmoid, and minus one to one for tanh. The MLP regressor class does not support output activation functions.

Scikit-Learn offers no GPU acceleration and fairly limited neural network features, which is why the book switches to PyTorch next, but it builds a standard MLP in a few lines.

The mean squared error is generally the right loss for regression. With many outliers in the training set, the mean absolute error can be preferable, or better, the Huber loss, which combines both. It is quadratic when the error is smaller than a threshold, delta, typically one, and linear when the error is larger. The linear part makes it less sensitive to outliers than the mean squared error, and the quadratic part makes it converge faster and more precisely than the mean absolute error. The MLP regressor supports only the mean squared error.

A typical regression MLP has one to five hidden layers, depending on the problem, with typically ten to one hundred neurons each, and one output neuron per target dimension. Hidden layers use ReLU. The output layer uses no activation, or ReLU or softplus for positive outputs, or sigmoid or tanh for bounded outputs. The loss is the mean squared error, or the Huber loss if there are outliers.

---

Classification MLPs. For binary classification, an MLP needs a single output neuron with the sigmoid activation function. Its output, between zero and one, is the estimated probability of the positive class, and the probability of the negative class is one minus that.

MLPs also handle multilabel binary classification. An email system could predict both whether each email is spam or ham, and whether it is urgent or not. That needs two output neurons, both using the sigmoid: one outputs the probability of spam, the other the probability of urgent. More generally, there is one output neuron per positive class. These probabilities do not need to add up to one, so the model can output any combination of labels: nonurgent ham, urgent ham, nonurgent spam, and even urgent spam, though that would probably be an error.

If each instance belongs to exactly one of three or more classes, such as the digits zero to nine, the network needs one output neuron per class, and the softmax activation function for the whole output layer. Softmax ensures that the estimated probabilities are all between zero and one and add up to one, since the classes are exclusive. This is multiclass classification.

Since the network predicts probability distributions, the cross-entropy loss, also called log loss, is generally a good choice in all three cases. A typical classification MLP has one to five hidden layers, depending on the task.

---

Classifying Fashion MNIST. Scikit-Learn's MLP classifier class is almost identical to the MLP regressor, except that its output layer uses softmax, it minimizes the cross-entropy loss instead of the mean squared error, and its score method returns accuracy instead of the R squared score.

The iris dataset is too simple for a neural network; a linear model would do just as well without risking overfitting. Fashion MNIST is a drop-in replacement for MNIST with the same format: seventy thousand grayscale images of twenty-eight by twenty-eight pixels, in ten classes. The images show fashion items instead of handwritten digits, so each class is much more diverse and the problem is significantly harder. A simple linear model reaches about ninety-two percent accuracy on MNIST, but only about eighty-three percent on Fashion MNIST.

The dataset is loaded from OpenML, with its string targets converted to integers. The first sixty thousand images form the training set and the last ten thousand the test set. Each image is an array of seven hundred and eighty-four pixel intensities from zero to two hundred and fifty-five. The ten class names are T-shirt or top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

The classifier has two hidden layers of three hundred and one hundred neurons, with early stopping. The pipeline uses a min-max scaler, which shrinks pixel intensities to the range zero to one and usually works better with the MLP classifier's defaults, such as its learning rate and weight initialization scale. A standard scaler would scale up nearly constant pixels, such as the almost always white edge pixels, to the same variance as every other pixel, giving them more importance than they deserve. For images, min-max scaling often works better.

The model reaches about eighty-nine point seven percent validation accuracy during training, then overfits a bit and ends at eighty-nine point two. Test accuracy is eighty-seven point one percent, not bad, though convolutional neural networks do better.

Training is slow because the hidden layers have many parameters. The first hidden layer alone has seven hundred and eighty-four times three hundred weights plus three hundred biases, two hundred and thirty-five thousand five hundred parameters. That flexibility brings a high risk of overfitting, especially with little data, which early stopping and L2 regularization help control.

On fifteen test images, every prediction is correct except one sneaker classified as a bag, and the model is one hundred percent confident in that wrong answer. Across all ten thousand test images, it is less than ninety-nine point nine percent confident on only sixteen, despite about ninety percent accuracy. Neural networks tend to be overconfident, especially when trained a bit too long, so their estimated probabilities deserve skepticism.

Classification targets can be class indices, or class probabilities, typically one-hot vectors with a one for the target class and zeros elsewhere. For an overconfident model, label smoothing can help: the target class's probability is reduced slightly, for example from one to zero point nine, and the remainder is spread evenly across the other classes.

Fine-tuning the hyperparameters, for example with a randomized search, can improve performance further, but the search space is large, so it helps to know roughly where to look.

---

Hyperparameter tuning guidelines. The flexibility of neural networks is also one of their main drawbacks: there are many hyperparameters to tweak. Beyond the choice of architecture, even a basic MLP has the number of layers, the number of neurons and the activation function in each layer, the weight initialization logic, the optimizer, its learning rate, the batch size, and more.

---

Number of hidden layers. For many problems, a single hidden layer gives reasonable results. In theory, an MLP with one hidden layer can model even the most complex functions, given enough neurons. But deep networks have much higher parameter efficiency than shallow ones. They can model complex functions with exponentially fewer neurons, so they reach much better performance with the same amount of training data.

Their layered structure allows them to reuse and compose features across levels. In a face classifier, the first layer may learn low-level features such as dots, arcs, and straight lines. The second may combine them into higher-level features such as squares and circles. The third may combine those into a mouth, an eye, or a nose. The top layer can then use these features to classify faces.

This hierarchy helps DNNs converge faster to a good solution and improves their ability to generalize to new datasets. A network for recognizing hairstyles can start from the lower layers of a network already trained to recognize faces: instead of random initialization, its first layers get the trained weights and biases. The new network then does not have to relearn the low-level structures common to most pictures, only the higher-level ones, such as hairstyles. This is transfer learning.

For many problems, one or two hidden layers work well. On MNIST, one hidden layer with a few hundred neurons easily reaches above ninety-seven percent accuracy, and two hidden layers with the same total number of neurons reach above ninety-eight percent in about the same training time. For more complex problems, hidden layers can be added until the network starts overfitting. Very complex tasks, such as large image classification or speech recognition, typically need dozens of layers, or even hundreds that are not fully connected, and huge amounts of data. Such networks are rarely trained from scratch: reusing parts of a pretrained state-of-the-art network for a similar task makes training much faster and needs much less data.

---

Number of neurons per hidden layer. The numbers of input and output neurons are set by the task. MNIST needs seven hundred and eighty-four inputs and ten output neurons.

Hidden layers used to be sized as a pyramid, with fewer neurons at each layer, on the idea that many low-level features coalesce into far fewer high-level ones. A typical MNIST network might have had three hidden layers of three hundred, two hundred, and one hundred neurons. This practice has been largely abandoned: using the same number of neurons in all hidden layers performs just as well in most cases, or better, and leaves only one hyperparameter to tune instead of one per layer. Depending on the dataset, it can help to make the first hidden layer a bit larger than the others.

As with layers, the number of neurons can be increased gradually until the network starts overfitting. Alternatively, a model can be built with slightly more layers and neurons than needed, using early stopping and other regularization to keep it from overfitting too much. Vincent Vanhoucke, a Waymo researcher and former Googler, called this the stretch pants approach: rather than searching for pants that fit perfectly, use large stretch pants that shrink to the right size.

This approach avoids bottleneck layers, which lack the capacity to model complex relationships and may not even preserve all the useful information from the inputs. PCA on Fashion MNIST needs one hundred and eighty-seven dimensions to preserve ninety-five percent of the variance, so a first hidden layer of two hundred neurons will not be a bottleneck. Too many neurons, though, mean more parameters, and more time and data to train. In general, adding layers gives more benefit than adding neurons per layer.

Bottleneck layers are not always bad. Limiting the size of the first hidden layers forces the network to keep only the most important dimensions, which can eliminate some noise, as long as it does not go too far. A bottleneck near the output layer forces the network to pack more useful information into less space in the preceding layers, learning good representations of the data. This can help generalization and is useful in itself for representation learning.

---

Learning rate. The learning rate is a hugely important hyperparameter. The optimal learning rate is generally about half the maximum learning rate, the rate above which training diverges.

To find a good learning rate, train the model for a few hundred iterations while increasing the learning rate from a very low value, such as ten to the minus five, to a very large one, such as ten, by multiplying it by a constant factor at each iteration. Plotted against the learning rate on a log scale, the loss drops at first, then shoots back up once the learning rate is too large. The optimal learning rate is often about ten times lower than the point where the loss starts to climb. The model is then reinitialized and trained normally with that rate.

To change the learning rate during training in Scikit-Learn, set the MLP's warm start hyperparameter to true, fit the model one batch at a time with the partial fit method, and update the learning rate at each iteration.

---

Batch size. The batch size can significantly affect performance and training time. Large batches are processed efficiently by hardware accelerators such as GPUs, so the training algorithm sees more instances per second. Many researchers and practitioners therefore recommend the largest batch size that fits in the GPU's memory, called VRAM.

Large batches can, however, cause training instabilities, especially with smaller models and at the start of training, and the resulting model may not generalize as well as one trained with small batches. Yann LeCun once joked on Twitter that mini-batches should never exceed thirty-two, citing a 2018 paper by Dominic Masters and Carlo Luschi, which concluded that small batches, from two to thirty-two, produced better models in less training time.

Other research points the other way. In 2017, papers by Elad Hoffer and colleagues and by Priya Goyal and colleagues showed that batches of up to eight thousand one hundred and ninety-two can give very short training times with no generalization gap, using techniques such as learning rate warmup, which starts with a small learning rate and ramps it up. So one strategy is a large batch size, possibly with warmup, switching to smaller batches if training is unstable or the final performance disappointing.

---

Other hyperparameters. With enough compute budget and time, two more hyperparameters can be tuned. Choosing a better optimizer than plain mini-batch gradient descent, and tuning its hyperparameters, can speed up training and sometimes improve performance. The activation function can also be changed: ReLU is a good default for all hidden layers, but replacing it can sometimes help.

The optimal learning rate depends on the other hyperparameters, especially the batch size, so after changing any hyperparameter, the learning rate should be tuned again.

Further guidance on tuning neural network hyperparameters can be found in a 2018 paper by Leslie Smith, in the Deep Learning Tuning Playbook by Google researchers, and in Andrew Ng's free e-book Machine Learning Yearning.

---

Summary.

Artificial neural networks are inspired by biological neurons but have diverged from them. They date back to 1943, went through cycles of enthusiasm and neglect, and now thrive thanks to huge datasets, GPUs, improved training algorithms, the Transformer, and a virtuous circle of funding and progress.

A threshold logic unit computes a weighted sum of its inputs plus a bias, then applies a step function. A perceptron is a single dense layer of such units, trained with a rule inspired by Hebbian learning. It can only learn linear decision boundaries, so it cannot solve XOR. A multilayer perceptron, made of stacked layers, can.

Backpropagation combines reverse-mode autodiff with gradient descent: a forward pass, an error measurement, a backward pass that applies the chain rule layer by layer to get every parameter's gradient, and a gradient descent step. Weights are initialized randomly to break symmetry. Activation functions need useful gradients, like the sigmoid, tanh, or the default, ReLU, and they must be nonlinear, or a deep network collapses into a single linear layer.

Regression MLPs use one output neuron per target value and the mean squared error or Huber loss. Classification MLPs use sigmoid outputs for binary or multilabel tasks and softmax for multiclass tasks, with the cross-entropy loss. Neural networks tend to be overconfident.

For tuning, start with one or two hidden layers and add more until overfitting, use the same number of neurons in each hidden layer, and avoid bottlenecks. The learning rate matters most, and should be retuned whenever other hyperparameters change. The batch size trades GPU efficiency against stability and generalization.
