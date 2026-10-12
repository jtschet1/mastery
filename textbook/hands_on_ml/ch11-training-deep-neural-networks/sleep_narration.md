# Chapter 11 — Training Deep Neural Networks, to fall asleep to

Chapter eleven. Training deep neural networks.

---

The problems of deep networks. Complex tasks, such as detecting hundreds of types of objects in high-resolution images, may need networks with dozens or hundreds of layers. Training such a deep neural network, or DNN, runs into four problems. Gradients can grow ever smaller or ever larger as they flow backward, which makes the lower layers very hard to train. There may not be enough labeled training data. Training may be extremely slow. And a model with millions of parameters risks severely overfitting, especially with few or noisy training instances.

---

Vanishing and exploding gradients. Backpropagation propagates the error gradient from the output layer down to the input layer, then updates every parameter with a gradient descent step. Gradients often shrink as they move down to the lower layers, so those layers' weights barely change and training never converges to a good solution. This is the vanishing gradients problem. In the opposite case, gradients grow until layers get huge updates and the algorithm diverges. This is the exploding gradients problem, most common in recurrent neural networks. More generally, deep networks suffer from unstable gradients, and different layers may learn at very different speeds.

This was one reason deep networks were mostly abandoned in the early 2000s. A 2010 paper by Xavier Glorot and Yoshua Bengio found some of the causes, including the combination of the sigmoid activation function with the popular initialization of the time, a normal distribution with mean zero and standard deviation one. With these, each layer's outputs have a much greater variance than its inputs, and the variance keeps growing layer after layer until the activation function saturates in the top layers. The sigmoid's mean of zero point five, rather than zero, makes this worse; the hyperbolic tangent, with mean zero, behaves slightly better. A saturated sigmoid, at zero or one, has a derivative extremely close to zero, so backpropagation has almost no gradient to send back, and what little there is gets diluted on the way down, leaving nearly nothing for the lower layers.

---

Glorot and He initialization. Glorot and Bengio argued that the signal must flow properly in both directions, forward for predictions and backward for gradients, without dying out or exploding. For that, each layer's outputs should have the same variance as its inputs, and gradients should have the same variance before and after flowing backward through a layer. Both can only hold if the layer has as many inputs as outputs, numbers called its fan-in and fan-out.

Their compromise, which works very well in practice, is to draw each layer's weights from a normal distribution with mean zero and a variance of one over the fan average, the mean of the fan-in and fan-out. Alternatively, a uniform distribution between minus r and plus r can be used, where r is the square root of the ratio of three to the fan average. This is Glorot initialization, or Xavier initialization. Using the fan-in instead of the fan average gives LeCun initialization, from the 1990s. It took over a decade for researchers to realize how important this trick is, and it is one of the tricks behind the success of deep learning.

Similar strategies exist for other activation functions, notably from a 2015 paper by Kaiming He and colleagues. They differ only in the scale of the variance and in using the fan average or the fan-in. Glorot initialization, variance one over the fan average, suits no activation, the hyperbolic tangent, the sigmoid, and softmax. He initialization, or Kaiming initialization, with a variance of two over the fan-in, suits ReLU and its variants. LeCun initialization, with variance one over the fan-in, suits SELU, preferably with a normal distribution. For a uniform distribution, r is the square root of the product of three and the variance.

---

Initialization in PyTorch. For historical reasons, PyTorch's linear module uses Kaiming uniform initialization scaled down by a factor of the square root of six, and initializes biases randomly, which is not optimal for any common activation function. Multiplying the weights by the square root of six right after creating the layer gives proper Kaiming initialization, and biases can be zeroed, since random biases bring no benefit. Clearer and less error-prone are the in-place functions in PyTorch's initialization module, such as Kaiming uniform and zeros. To initialize every linear layer of a model, a small function checks whether a module is a linear layer and initializes it, and the model's apply method runs that function on every submodule.

The initialization module also offers orthogonal initialization, proposed by Andrew Saxe and colleagues in 2014. Orthogonal matrices preserve norms, so the magnitude of the inputs carries through to the outputs, which keeps the variance stable and prevents vanishing or exploding at least at the start of training. It is less common, but can work well in recurrent networks and generative adversarial networks.

In a classifier, scaling down the output layer's initial weights, for example by a factor of ten, is generally a good idea. The initial logits are then closer together, so the model starts out less confident, which avoids extreme losses and huge gradients that make the weights bounce around at the start of training.

---

Better activation functions. Glorot and Bengio also found that unstable gradients came partly from a poor choice of activation function. People had assumed that since biological neurons use something close to a sigmoid, it must be a good choice. ReLU works much better in deep networks, mostly because it does not saturate for positive values, and because it is very fast to compute.

ReLU suffers from dying ReLUs: some neurons stop outputting anything but zero, sometimes half of a network, especially with a large learning rate. A neuron dies when its weighted sum of inputs plus bias is negative for every training instance. It then outputs only zeros, and since ReLU's gradient is zero for negative inputs, gradient descent no longer affects it.

---

Leaky ReLU. The leaky ReLU is the maximum of alpha times z and z, where the hyperparameter alpha is the slope for negative inputs. Because of this slope, leaky ReLUs never truly die: they can fall into a long coma but may wake up. A 2015 paper by Bing Xu and colleagues found that leaky variants always outperformed strict ReLU, and that a large leak, alpha equal to zero point two, seemed better than a small one of zero point zero one. Randomized leaky ReLU, or RReLU, picks alpha randomly within a range during training and fixes it to an average at test time; it performed fairly well and seemed to act as a regularizer. Parametric leaky ReLU, or PReLU, learns alpha by backpropagation. It strongly outperformed ReLU on large image datasets, but risks overfitting smaller ones.

PyTorch has modules for all three. They should be used with Kaiming initialization, with the variance scaled down by a factor of one plus alpha squared; PyTorch's Kaiming functions accept alpha to make this adjustment.

ReLU, leaky ReLU, and PReLU are not smooth: their slope changes abruptly at zero, which can make gradient descent bounce around the optimum and slow convergence.

---

ELU and SELU. The exponential linear unit, or ELU, proposed in 2015 by Djork-Arné Clevert and colleagues, outperformed all ReLU variants in their experiments. For negative inputs it equals alpha times the exponential of z minus one, and for positive inputs it equals z. It takes negative values, which brings the average output closer to zero and helps with vanishing gradients; alpha, usually one, is the opposite of the value it approaches for very negative inputs. It has a nonzero gradient for negative inputs, so neurons do not die. And with alpha equal to one, it is smooth everywhere. Its main drawback is that the exponential makes it slower to compute than ReLU, so an ELU network is a bit slower at test time.

The scaled ELU, or SELU, introduced in 2017 by Günter Klambauer and colleagues, is about one point zero five times ELU, with alpha about one point six seven. In a plain stack of dense layers, a multilayer perceptron or MLP, where every hidden layer uses SELU, the network self-normalizes: each layer's output tends to keep a mean of zero and a standard deviation of one, which solves unstable gradients. The conditions are strict. Inputs must be standardized, hidden layers must use LeCun normal initialization, the architecture must be a plain MLP without recurrent layers or skip connections, and regularization such as L1 or L2, batch normalization, layer normalization, max-norm, and regular dropout cannot be used. So SELU never gained much traction, and other functions now tend to outperform it.

---

GELU, Swish, SwiGLU, Mish, and ReLU squared. The Gaussian error linear unit, or GELU, introduced in 2016 by Dan Hendrycks and Kevin Gimpel, is a smooth variant of ReLU. It equals z times the probability that a value drawn from a standard normal distribution is lower than z. It approaches zero for very negative inputs and z for very positive ones, but unlike the earlier functions it is neither convex nor monotonic: it dips to a low point of about minus zero point one seven near z equals minus zero point seven five, then rises. This more complex shape, curved everywhere, may explain why it often outperforms all the previous functions, especially on complex tasks. It costs more compute, but z times the sigmoid of one point seven zero two times z approximates it well and is much faster.

The GELU paper also introduced the sigmoid linear unit, or SiLU, which is z times the sigmoid of z. In 2017, Prajit Ramachandran and colleagues rediscovered it through an automated search for activation functions and named it Swish, and in their tests it beat every other function, including GELU. They generalized it with a hyperparameter beta that scales the sigmoid's input, so GELU is approximately Swish with beta equal to one point seven zero two. Beta can be tuned or made trainable, usually one per model or per layer.

SwiGLU sends the inputs through Swish and, in parallel, through a linear layer, then multiplies the two outputs itemwise. In practice, the previous layer's output size is doubled and split in two, and Swish of the first half is multiplied by the second half. It is a variant of the gated linear unit, or GLU, from Facebook researchers in 2016. The multiplication is a gating mechanism, which allows the model to turn off or amplify specific features. SwiGLU is very common in modern transformers.

Mish, introduced in 2019 by Diganta Misra, is z times the hyperbolic tangent of the softplus of z, where softplus is the log of one plus the exponential of z. It is smooth, nonconvex, and nonmonotonic, and its author found it slightly outperformed even Swish and GELU. It closely matches Swish for negative inputs and GELU for positive ones.

ReLU squared, found in 2021 by an automated architecture search at Google to improve large transformers, is the square of ReLU. It keeps ReLU's simplicity, efficiency, sparse output, and lack of positive saturation, while having smooth gradients at zero. It often outperforms other functions, especially in sparse models, but training can be less stable, partly due to sensitivity to outliers and to dying ReLUs.

---

Choosing an activation function. ReLU remains a good default for most tasks: it is often as good as fancier functions, very fast, and specially optimized by many libraries and accelerators. Swish is probably a better default for complex tasks, with a learnable beta for the most complex ones. Mish and SwiGLU may be slightly better but need more compute. When latency matters most, leaky ReLU, parametric leaky ReLU, or ReLU squared are good choices.

PyTorch provides GELU, Mish, and Swish, the last named SiLU. SwiGLU and ReLU squared take a line or two to build from existing functions. PyTorch also offers simplified approximations whose names start with hard, such as hard sigmoid and hard swish, which are much faster, often more stable, and common on mobile devices.

---

Batch normalization. Kaiming initialization with ReLU reduces unstable gradients at the start of training, but does not guarantee they will not return. Batch normalization, or BN, proposed in 2015 by Sergey Ioffe and Christian Szegedy, adds an operation just before or after each hidden layer's activation function. It zero-centers and normalizes each input, then scales and shifts the result with two learned parameter vectors per layer, so the model learns the optimal scale and mean of each input. A BN layer placed first in the network can often replace standardizing the training set.

The statistics are estimated over the current mini-batch, hence the name. The algorithm computes each input's mean over the mini-batch, then its variance. It subtracts the mean from each input and divides by the square root of the variance plus epsilon, a tiny smoothing term, typically ten to the minus five, that avoids division by zero. Finally, it multiplies each normalized input by the output scale vector, gamma, and adds the output shift vector, beta.

At test time, predictions may be made one instance at a time, and even a batch may be too small or not independent and identically distributed, so batch statistics are unreliable. Final statistics for the whole training set are needed instead. Most implementations, including PyTorch's, estimate them during training with an exponential moving average of the batch means and variances.

So each BN layer has four parameter vectors. Gamma and beta are learned by backpropagation. The final mean and final variance are estimated with moving averages during training, and used in place of the batch statistics once the model is switched to evaluation mode.

---

Benefits and costs of batch normalization. Ioffe and Szegedy found that BN considerably improved every deep network they tried, with a huge improvement on ImageNet, a large image database commonly used to evaluate computer vision systems. Vanishing gradients were so reduced that even the hyperbolic tangent and the sigmoid could be used. Networks were much less sensitive to initialization, and much larger learning rates sped up learning. On a state-of-the-art image model, BN reached the same accuracy in fourteen times fewer training steps, and an ensemble of batch-normalized networks beat the accuracy of human raters on ImageNet. BN also acts as a regularizer, reducing the need for techniques such as dropout.

BN adds complexity, and a runtime penalty from the extra computation in each layer. After training, a BN layer can often be fused into the previous layer. If that layer computes the inputs times W plus b, the BN layer computes gamma times that result minus the mean, divided by the standard deviation, plus beta. Folding those operations into new weights and a new bias makes the previous layer produce normalized outputs directly, so the BN layer can be removed. PyTorch's optimize for inference function does this.

Each epoch takes longer with BN, but convergence is much faster, so overall wall time is usually shorter.

---

Batch normalization in PyTorch. A one-dimensional batch norm layer, given its number of inputs, is added before or after each hidden layer's activation function, and optionally first, after flattening, to normalize the inputs. In a small two-layer Fashion MNIST classifier the effect is modest, but in deeper networks it can make a tremendous difference. Because BN behaves differently during training and evaluation, switching between training mode and evaluation mode is critical, and forgetting it is one of the most common mistakes.

A BN layer's two parameters, weight and bias, are gamma and beta. Its three buffers are the running mean, the running variance, and a count of batches seen during training.

The BN paper's authors favored placing BN before the activation function. This is debated and seems task-dependent, so both are worth trying. With BN before the activation, the preceding linear layers can drop their bias terms, since BN already has one offset per input.

The main hyperparameter to adjust is momentum, used to update the moving averages: the new running average equals the batch value times the momentum, plus the old average times one minus the momentum. A good value is close to zero, such as zero point zero one or zero point zero zero one, smaller for smaller batches. The default of zero point one suits large batches but not batches of thirty-two or sixty-four. Conventionally, momentum means the weight of the old average, but PyTorch's BN layers use the opposite meaning, while its optimizers use the conventional one.

---

Batch norm in one, two, and three dimensions. One-dimensional batch norm expects batches of shape batch size by number of features, like a linear layer, which is why images are flattened first. Two-dimensional batch norm works directly on image batches of shape batch size by channels by height by width, computing statistics across the batch and both spatial dimensions. All pixels in a channel share the same statistics, and the layer has one weight and one bias per channel. This generally works better for images. Three-dimensional batch norm handles volumes, such as computed tomography scans.

One-dimensional batch norm also handles sequences, but PyTorch stores sequences as batch size by sequence length by features, while the BN layer expects features before sequence length, so the last two dimensions must be swapped first.

BN has become one of the most used layers, especially in convolutional networks, often omitted from architecture diagrams because it is assumed. But an instance's statistics are biased by the other samples in its batch, which hurts with small batches, and BN struggles with architectures such as recurrent networks. So it is increasingly replaced by layer normalization.

---

Layer normalization. Layer normalization, or LN, introduced in 2016 by Jimmy Lei Ba and colleagues, normalizes across the feature dimensions instead of the batch dimension. First used mostly in recurrent networks, it now appears in convolutional networks, transformers, diffusion models, and more. It computes statistics on the fly for each instance independently, so it behaves the same in training and testing and needs no moving averages. Like BN, it learns a scale and an offset per input feature.

PyTorch's layer norm module is created with the sizes of the dimensions to normalize over, which must be the last dimensions of the input. For color images, normalizing over height and width normalizes each channel separately, while including the channel dimension normalizes all channels at once, which is what most vision architectures using LN do.

---

Gradient clipping. Gradient clipping caps gradients during backpropagation so they never exceed a threshold. It mitigates exploding gradients and is mostly used in recurrent networks, where BN is tricky. In PyTorch, a clipping function is called on the model's parameters at each iteration, after backward and before the optimizer's step. Clipping by norm rescales each gradient vector whose norm exceeds a maximum, typically one. Clipping by value clips each component to between minus and plus a threshold.

Clipping by value can change the gradient's direction: a gradient of zero point nine and one hundred, clipped to one, becomes zero point nine and one, pointing roughly along the diagonal instead of along the second axis. It still works quite well in practice. Clipping by norm keeps the direction but nearly erases the first component. The better choice depends on the dataset.

---

Reusing pretrained layers. Rather than train a large DNN from scratch, it is generally better to find a network trained on a similar task and reuse most of its layers except the top ones. This is transfer learning. It speeds up training considerably and needs much less data. A network that classifies one hundred categories of animals, plants, vehicles, and everyday objects can be partly reused to classify types of vehicles. Inputs may need resizing to the original model's expected size, and transfer works best when inputs share low-level features: a network trained on phone pictures will not help much with satellite or medical images.

The output layer should usually be replaced. Upper hidden layers are less likely to be useful than lower ones, since high-level features differ more between tasks. The more similar the tasks, the more layers to reuse, starting from the bottom.

The procedure is to freeze all reused layers by turning off their gradient tracking, train, and evaluate. Then unfreeze one or two top hidden layers and see if performance improves. More data allows unfreezing more layers. Lowering the learning rate when unfreezing protects the reused weights. With little data and poor results, top hidden layers can be dropped; with plenty of data, they can be replaced, and more layers added.

---

Transfer learning with PyTorch. In the example, model A was trained on Fashion MNIST without the pullover and T-shirt classes, and reached about ninety-two percent accuracy on the remaining eight. Task B is a binary T-shirt versus pullover classifier with only twenty labeled images, and a model with the same architecture trained from scratch reaches seventy-one point six percent.

All of model A's layers except the output layer are deep-copied, so the original model is unaffected, and a new sequential model is built from them plus a new single-output layer. The new output layer starts random and makes large errors, whose large gradients could wreck the reused weights, so the reused layers are frozen for the first few epochs. Since task B is binary, the loss is the binary cross-entropy with logits loss. After a few epochs, the reused layers are unfrozen, the learning rate is reduced, and training continues. The result is ninety-two point five percent test accuracy.

The book admits this result was cherry-picked. Many configurations were tried until one showed a strong gain; other classes or random seeds shrink, erase, or reverse it. This is torturing the data until it confesses. Reporting only the best of many attempts, which may be pure luck, is called p-hacking, and it is part of why so many scientific results cannot be reproduced. Transfer learning works poorly with small dense networks, presumably because small networks learn few patterns and dense networks learn very specific ones. It works best with deep convolutional networks and transformers.

---

Unsupervised pretraining. With little labeled data and no model trained on a similar task, the first step is to gather more labels. Failing that, unlabeled data, which is often cheap, can be used to train an unsupervised model such as an autoencoder. Its lower layers are reused, an output layer for the task is added, and the network is fine-tuned with supervised learning on the labeled examples.

Geoffrey Hinton's team used this in 2006, which revived neural networks and led to the success of deep learning. Until 2010, unsupervised pretraining with restricted Boltzmann machines, or RBMs, was the norm, done greedily one layer at a time: train a single-layer model, freeze it, add a layer, train again, and repeat. Today the full unsupervised model, usually an autoencoder or diffusion model, is trained in one shot. It remains a good option for a complex task with no reusable model, little labeled data, and plenty of unlabeled data.

---

Pretraining on an auxiliary task. Another option is to train a network on an auxiliary task with easily obtained or generated labels, then reuse its lower layers. For face recognition with only a few pictures per person, a network can learn from a public dataset of millions of pictures of people, such as VGGFace2, to detect whether two pictures show the same person. Its lower layers become good face feature detectors. Scraping random people's pictures from the web instead would probably be illegal, because of copyright, sites' terms of service, and consent laws in over forty countries.

For natural language processing, or NLP, labels can be generated from a large text corpus by masking random words and training a model to predict them. A model that does this well already knows a lot about language and can be fine-tuned for the real task. This is essentially how large language models are trained. Generating labels from the data itself and then training with supervised techniques is called self-supervised learning.

---

Momentum optimization. Besides good initialization, good activation functions, normalization, and reused layers, a faster optimizer than plain gradient descent gives another huge speedup. The most popular are momentum, Nesterov accelerated gradient, AdaGrad, RMSProp, and Adam with its variants.

Momentum optimization, proposed by Boris Polyak in 1964, is like a bowling ball rolling down a gentle slope: it starts slowly but picks up speed until friction limits it. Regular gradient descent subtracts the learning rate times the current gradient and ignores earlier gradients, so it never picks up speed and crawls when the gradient is tiny.

Momentum keeps a momentum vector. At each iteration, the vector is multiplied by the momentum coefficient, beta, and the learning rate times the gradient is subtracted from it; then the vector is added to the weights. The gradient acts as an acceleration rather than a speed. Beta acts as friction: zero means high friction, one means no friction, and the typical value is zero point nine.

With a constant gradient, the largest update is the gradient times the learning rate times one over one minus beta, so with beta at zero point nine, momentum goes ten times faster than gradient descent. In practice the gain is smaller, but momentum escapes plateaus much faster. When inputs have very different scales, the cost function is an elongated bowl, and momentum rolls down the valley faster and faster where gradient descent crawls. This helps a lot in deep networks without normalization, whose upper layers often get inputs of very different scales, and it can roll past local optima. It can overshoot and oscillate before settling, which is why some friction helps. In PyTorch, it is enabled by setting the momentum of the stochastic gradient descent optimizer, or SGD, and zero point nine usually works well.

---

Nesterov accelerated gradient. Nesterov accelerated gradient, or NAG, proposed by Yurii Nesterov in 1983, is almost always faster than regular momentum. It measures the gradient slightly ahead, at the current weights plus beta times the momentum vector. Since momentum generally points toward the optimum, that gradient is slightly more accurate, and the small gains add up. When momentum pushes the weights across a valley, the look-ahead gradient pushes back toward the bottom, which reduces oscillations. In PyTorch, it is enabled with the SGD optimizer's Nesterov option.

---

AdaGrad. In an elongated bowl, gradient descent first goes down the steepest slope, which does not point at the global optimum. AdaGrad corrects its direction earlier by scaling down the gradient along the steepest dimensions. First, it adds the square of each gradient component to an accumulator vector, s, which grows fastest along steep dimensions. Second, it takes a gradient descent step in which each gradient component is divided by the square root of its accumulator plus a tiny smoothing term.

In effect, the learning rate decays faster for steep dimensions than for gentle ones. This is an adaptive learning rate. It points updates more directly toward the optimum and needs less tuning of the learning rate. AdaGrad works well on simple quadratic problems, but with neural networks it often stops too early, because the learning rate shrinks so much that it stops before reaching the optimum. It should not be used for deep networks, but understanding it helps with the other adaptive optimizers.

---

RMSProp. RMSProp, short for root mean square propagation, created by Geoffrey Hinton and Tijmen Tieleman in 2012, fixes AdaGrad's early stop by accumulating only recent gradients. In its first step, the accumulator is multiplied by a decay rate, alpha, before adding one minus alpha times the squared gradient. The decay rate is typically zero point nine and rarely needs tuning. Except on very simple problems, RMSProp almost always beats AdaGrad, and it was many researchers' favorite optimizer until Adam.

---

Adam. Adam, short for adaptive moment estimation, combines momentum and RMSProp. It tracks an exponentially decaying average of past gradients, the first moment, and an exponentially decaying average of past squared gradients, the uncentered second moment.

At each iteration, Adam updates the gradient average, m, weighting the old value by beta one, and the squared gradient average, s, weighting the old value by beta two. It then corrects both for their bias toward zero early in training, since they start at zero, by dividing m by one minus beta one to the power of the iteration number, and s by one minus beta two to the power of the iteration number. Finally, it updates the weights by the learning rate times the corrected m, divided by the square root of the corrected s plus a smoothing term.

Beta one plays the role of momentum's beta, and beta two the role of RMSProp's decay rate. The defaults are zero point nine for beta one, zero point nine nine nine for beta two, and ten to the minus eight for the smoothing term. Because Adam adapts its learning rate, the default learning rate of zero point zero zero one often works, which makes Adam even easier to use than gradient descent.

---

AdaMax, NAdam, and AdamW. AdaMax, from the Adam paper, replaces the L2 norm of the time-decayed gradients, which Adam uses to scale updates, with the maximum. Its accumulator becomes the larger of beta two times its previous value and the gradient's absolute value. It can be more stable than Adam on some datasets, but Adam is generally better.

NAdam is Adam plus the Nesterov trick, so it often converges slightly faster. Its author, Timothy Dozat, found that it generally beats Adam but is sometimes beaten by RMSProp.

AdamW, by Ilya Loshchilov and Frank Hutter, integrates weight decay, which multiplies the weights by a factor such as zero point nine nine at each iteration. With SGD, weight decay is mathematically equivalent to L2 regularization, but with Adam it is not, and combining Adam with L2 regularization often gives models that generalize worse than SGD. AdamW combines Adam with weight decay properly, and its weight decay hyperparameter is usually worth tuning.

Adaptive optimizers, meaning RMSProp, Adam, AdaMax, NAdam, and AdamW, often converge fast to a good solution, but a 2017 paper by Ashia Wilson and colleagues showed they can generalize poorly on some datasets. When a model disappoints, NAG is worth trying.

---

Second-order methods, sparse models, and choosing an optimizer. All these optimizers use only first-order partial derivatives, the Jacobians, which measure slopes. Algorithms based on second-order derivatives, the Hessians, which measure curvature, are hard to apply to deep networks: there are n squared second-order derivatives per output, n being the number of parameters, so they often do not fit in memory and are too slow to compute. Stochastic methods such as Shampoo approximate second-order information efficiently, and Shampoo is available in the PyTorch Optimizer library.

These optimizers produce dense models. For a faster, smaller sparse model, the tiny weights can be zeroed after training with PyTorch's pruning utilities, which can also remove whole neurons, channels, or layers. Strong L1 regularization during training helps, by pushing as many weights as possible to zero.

Overall, plain SGD converges slowly but to a good solution. Momentum and NAG converge faster, also to a good solution. AdaGrad converges fast but stops too early. RMSProp, Adam, AdaMax, NAdam, and AdamW converge fast, to a solution that is average or good depending on the dataset.

---

Learning rate scheduling. If the learning rate is too high, training diverges. If too low, training is very slow and may get stuck in a local optimum. If fairly high, training progresses fast but ends up dancing around the optimum. A good constant rate works but is slow. It is better to start fairly high and reduce the learning rate toward the end or whenever progress stops. Strategies for changing the learning rate during training are called learning schedules, and PyTorch's scheduler module implements the common ones.

Exponential scheduling multiplies the learning rate by a factor, gamma, at regular intervals, typically each epoch, so after n epochs it equals the initial rate times gamma to the power of n. Gamma should be below one but close to it. With gamma equal to zero point nine, the rate falls to about thirty-five percent after ten epochs and twelve percent after twenty. The scheduler wraps the optimizer, and the training loop calls the scheduler's step method at the end of each epoch.

---

Cosine annealing. Cosine annealing follows a cosine curve from a maximum learning rate down to a minimum. At epoch t, the learning rate equals the minimum, plus half the gap between maximum and minimum times one plus the cosine of pi times t over the total number of epochs. The rate stays fairly high for most of training and approaches the minimum near the end, and it generally performs better than exponential scheduling. Its drawback is two new hyperparameters, the number of epochs and the minimum rate, which are hard to choose in advance.

---

Performance scheduling. Performance scheduling, also called adaptive scheduling, tracks a metric, typically the validation loss, and multiplies the learning rate by a factor when the metric stops improving for a while. PyTorch implements it as the reduce learning rate on plateau scheduler. Its mode must be set to max for a metric to maximize, such as accuracy. Its patience, the number of epochs without improvement before reducing, defaults to ten. Its factor defaults to zero point one. The training loop evaluates the metric on the validation set at the end of each epoch and passes it to the scheduler's step method. The author generally prefers this schedule.

---

Warming up the learning rate. Starting at the maximum learning rate can make gradient descent bounce around randomly, neither exploding nor progressing. This happens with sensitive models, such as recurrent networks, or very large batches. Warm-up starts the learning rate near zero and raises it gradually over a few epochs. The loss landscape can be like the Himalayas, high and full of spikes: a high learning rate jumps from peak to peak, while a small one walks down into flatter land, where a large rate can then be used.

In PyTorch, a linear scheduler can raise the rate from ten percent to one hundred percent of its base value over three epochs: ten, forty, and seventy percent in the first three epochs, then one hundred. A lambda scheduler wraps any custom function. The warm-up scheduler steps at the start of each epoch, and other schedulers are paused during warm-up.

Raising the learning rate mid-training can help escape a local optimum or a plateau. The simplest way is cosine annealing with warm restarts.

---

Cosine annealing with warm restarts. Introduced in 2016 by Ilya Loshchilov and Frank Hutter, this schedule repeats cosine annealing again and again. Each time the learning rate jumps back up, gradient descent can escape local optima and plateaus. The authors recommend a short first round and doubling the round length each time, which allows many quick explorations early and careful optimization later.

---

One-cycle scheduling. One-cycle scheduling, introduced in 2018 by Leslie Smith, raises the learning rate linearly from an initial value to a maximum halfway through training, lowers it linearly back to the initial value during the second half, and over the last few epochs drops it by several orders of magnitude. The initial value is usually ten times lower than the maximum. Momentum does the reverse: from a high value such as zero point nine five down to zero point eight five in the first half, then back up for the rest. On the CIFAR10 dataset, this reached ninety-one point nine percent validation accuracy in one hundred epochs, versus ninety point three percent in eight hundred epochs with a standard approach. This was called super-convergence.

One-cycle is a good default, though the author often has more luck with performance scheduling. Warm-up helps with instability at the start, and warm restarts help with plateaus.

---

L1 and L2 regularization. With tens of thousands to billions of parameters, deep networks can fit almost anything, so they are prone to overfitting. Early stopping is one common remedy, and batch and layer normalization also regularize.

L2 regularization keeps weights small, and L1 regularization makes many weights exactly zero. With SGD, L2 regularization equals weight decay, set through the optimizer's weight decay argument, for example ten to the minus four. With Adam, AdamW with weight decay works better.

Weight decay applies to every parameter, including biases and normalization parameters, where it adds little and can hurt. One fix is to compute the L2 penalty manually, summing the squares of only the chosen parameters and adding it, times a factor, to the loss. Another is parameter groups: the optimizer gets several groups, each with its own hyperparameters, such as one group with weight decay and one without. Parameter groups also allow different learning rates per group, common in transfer learning so new layers learn faster than reused ones. PyTorch has no L1 helper, so the L1 penalty, the sum of absolute values, is added to the loss manually.

---

Dropout. Dropout was proposed by Geoffrey Hinton and colleagues in 2012 and detailed by Nitish Srivastava and colleagues in 2014. Many state-of-the-art networks use it, typically for a one to two percent accuracy gain. At ninety-five percent accuracy, a two percent gain cuts the error rate by almost forty percent.

At every training step, every neuron, including inputs but never outputs, has a probability p of being temporarily dropped, meaning ignored for that step. The dropout rate p is typically ten to fifty percent: twenty to thirty in recurrent networks, forty to fifty in convolutional networks. After training, nothing is dropped.

It works because neurons cannot co-adapt with their neighbors and must be useful on their own. They cannot rely on a few inputs and must attend to all of them, so they become less sensitive to slight input changes, and the network generalizes better. Also, with N droppable neurons there are two to the power of N possible networks, so each training step trains a practically unique network. The final network acts like an averaging ensemble of all of them, though they share weights.

Higher layers benefit most, so dropout is often applied only to the top one to three hidden layers. Output neurons should never be dropped.

One technical detail matters. With p at seventy-five percent, only a quarter of the neurons are active during training, so after training each neuron would get four times more input. So during training, the inputs are divided by the keep probability, one minus p. PyTorch's dropout layer does this in training mode and does nothing in evaluation mode, so switching modes matters, as with batch normalization.

Because dropout is only active during training, training and validation losses can look similar even when the model overfits, so the training loss should be evaluated without dropout. If the model overfits, the rate can be raised; if it underfits, lowered. Dropout slows convergence considerably, but properly tuned it often gives a better model. Self-normalizing SELU networks should use alpha dropout instead, which preserves the mean and standard deviation of its inputs.

---

Monte Carlo dropout. A 2016 paper by Yarin Gal and Zoubin Ghahramani connected dropout networks to approximate Bayesian inference, giving dropout a solid mathematical basis. It also introduced Monte Carlo dropout, or MC dropout, which boosts any trained dropout model without retraining it, and gives a much better measure of the model's uncertainty.

MC dropout keeps dropout active during inference, makes many predictions, and averages them. In PyTorch, the model is switched to evaluation mode, then its dropout layers are switched back to training mode. Each new image is repeated one hundred times in a batch, the model predicts logits for every copy, softmax turns them into probabilities, and the probabilities are averaged per image. Averaging the logits before applying softmax would be faster, but it does not properly reflect the model's uncertainty and tends to make it overconfident.

MC dropout makes probability estimates more reliable, so the model is less likely to be confidently wrong, such as a self-driving car confidently ignoring a stop sign. The standard deviation of each class probability across samples measures uncertainty. In the example, a ninety-nine percent ankle boot estimate has a standard deviation of zero point zero two, so the model is really saying something like over ninety-five percent. A risk-sensitive system might act only on predictions with a high probability and a low standard deviation.

More samples give more accurate predictions and uncertainty estimates but slower inference, with diminishing returns. To train an MC dropout model from scratch, a custom dropout module that always applies dropout is more robust than switching modes by hand.

---

Max-norm regularization. Max-norm regularization constrains each neuron's incoming weights so that their L2 norm stays at or below a maximum, r. Lowering r increases regularization, and max-norm can also help with unstable gradients when normalization layers are not used. It is applied after each training step rather than through the loss: inside a no grad context, each neuron's weight norm is computed, one per row of a linear layer's weight matrix, clamped to the maximum, and the weights are rescaled by the clamped norm divided by the actual norm, with a small epsilon to avoid dividing by zero.

---

Practical guidelines. A default configuration that works in most cases without much tuning: He initialization; ReLU for shallow networks and Swish for deep ones; no normalization for shallow networks and batch or layer normalization for deep ones; early stopping, plus weight decay if needed; Nesterov accelerated gradient or AdamW; and performance scheduling or one-cycle. These are not hard rules. A pretrained network for a similar problem should be reused when available.

A sparse model calls for L1 regularization, and possibly pruning small weights after training. A low-latency model calls for fewer layers, a fast activation such as ReLU, leaky ReLU, or hard swish, normalization layers folded into the previous layers, sparsity, and reduced float precision, from thirty-two bits to sixteen or eight. For risk-sensitive applications, or when latency matters little, MC dropout boosts performance and gives reliable probability and uncertainty estimates.

---

Summary.

Deep networks suffer from vanishing and exploding gradients. Glorot, He, and LeCun initialization keep signal and gradient variance stable across layers, using variances scaled by the fan-in or fan average.

ReLU does not saturate for positive inputs but can die. Leaky ReLU, ELU, SELU, GELU, Swish, SwiGLU, Mish, and ReLU squared address this in various ways. ReLU is a good default, Swish a better one for complex tasks.

Batch normalization standardizes each layer's inputs over the mini-batch, then rescales and shifts them with learned parameters, using moving averages at inference. Layer normalization normalizes each instance across its features and is replacing it. Gradient clipping limits exploding gradients.

Transfer learning reuses lower layers of a network trained on a similar task, frozen first and then fine-tuned. Unsupervised pretraining and auxiliary tasks help when labels are scarce.

Momentum and NAG accelerate gradient descent. AdaGrad, RMSProp, and Adam adapt the learning rate per parameter, and AdamW handles weight decay properly. Learning rate schedules, such as exponential, cosine annealing, performance scheduling, warm-up, warm restarts, and one-cycle, speed up training and improve models.

Regularization for deep networks includes early stopping, L1 and L2 regularization or weight decay, dropout, MC dropout, which also estimates uncertainty, and max-norm.
