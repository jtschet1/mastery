# Chapter 18 — Autoencoders, GANs, and Diffusion Models, to fall asleep to

Chapter eighteen. Autoencoders, GANs, and diffusion models.

---

Three families of generative models. Autoencoders are neural networks that learn dense representations of their input data, called latent representations or codings, without any supervision. The codings usually have far fewer dimensions than the input, so autoencoders are useful for dimensionality reduction, especially for visualization. They also act as feature detectors, can be used for unsupervised pretraining of deep networks, and are commonly used for anomaly detection. Some autoencoders are generative models: trained on pictures of faces, they can generate new faces.

Generative adversarial networks, or GANs, also generate data, and they can produce faces so convincing that it is hard to believe the people do not exist, as with the StyleGAN architecture. GANs have been used for super resolution, which means increasing an image's resolution, colorization, image editing, turning sketches into photorealistic images, predicting the next frames of a video, augmenting datasets, generating other types of data such as audio and time series, and finding weaknesses in other models. Since the early 2020s, GANs have been largely replaced by diffusion models, which generate more diverse, higher-quality images and are much easier to train. Diffusion models are much slower to run, though, so GANs remain useful when generation must be fast.

All three are unsupervised, learn latent representations, can be used as generative models, and share many applications, but they work very differently. An autoencoder learns to copy its inputs to its outputs, under constraints such as small codings or noisy inputs that prevent trivial copying, so the codings are a byproduct of learning the identity function under constraints. A GAN has two networks: a generator that tries to produce data resembling the training data, and a discriminator that tries to tell real data from fake. They compete during training, which is called adversarial training, like a counterfeiter trying to make realistic money and a police investigator trying to spot fakes. A diffusion model is trained to gradually remove noise from an image. Starting from pure random noise and running it repeatedly makes a high-quality image emerge, similar to the training images but not identical.

---

Efficient data representations. Of two number sequences, one short and random, the other a longer list of the even numbers from fifty down to fourteen, the longer one is easier to memorize, because only the pattern and the two endpoints need to be stored. If memorizing long sequences were easy, there would be no reason to look for a pattern. Constraining an autoencoder works the same way: it pushes the network to find and exploit patterns in the data.

In the early 1970s, William Chase and Herbert Simon studied the link between memory, perception and pattern matching. Expert chess players could memorize the positions of all the pieces after looking at a board for five seconds, but only when the positions came from real games, not when the pieces were placed randomly. Their memory was not better than anyone else's; they recognized chess patterns, which let them store the information efficiently.

An autoencoder does something similar. It always has two parts. The encoder, or recognition network, converts the inputs to a latent representation. The decoder, or generative network, converts that representation back to outputs. In a simple autoencoder built as a multilayer perceptron with three inputs, the output layer must also have three neurons, because the outputs, called reconstructions, try to reproduce the inputs. The cost function always includes a reconstruction loss that penalizes the model when the reconstructions differ from the inputs.

When the internal representation has fewer dimensions than the input, for example two instead of three, the autoencoder is called undercomplete. It cannot simply copy its inputs into the codings, yet it must output a copy of them, so it is forced to compress the data, learning the most important features and dropping the unimportant ones.

---

Principal component analysis with a linear autoencoder. If an autoencoder uses only linear activations and the mean squared error as its cost function, it ends up performing principal component analysis. The book builds one in PyTorch from two linear layers with no activation functions: an encoder projecting three dimensions down to two, and a decoder projecting back up to three. It is trained on a generated three-dimensional dataset, with the inputs also used as the targets. Gradient descent must find the two-dimensional plane closest to the data, just as principal component analysis would. After training, running the encoder on the data gives its two-dimensional projection, which preserves as much of the variance as possible.

An autoencoder can be seen as a form of self-supervised learning: it uses a supervised technique with automatically generated labels, here simply equal to the inputs.

---

Stacked autoencoders. Autoencoders with several hidden layers are called stacked, or deep, autoencoders. More layers let them learn more complex codings, but the autoencoder must not be too powerful. An encoder powerful enough to map each input to a single arbitrary number, with a decoder that learns the reverse mapping, would reconstruct the training data perfectly but learn no useful representation, and would be unlikely to generalize.

A stacked autoencoder is usually symmetrical around the central coding layer, like a sandwich. For Fashion MNIST, it might have seven hundred eighty-four inputs, a hidden layer of one hundred twenty-eight neurons, a central coding layer of thirty-two, another hidden layer of one hundred twenty-eight, and an output layer of seven hundred eighty-four. All hidden layers need an activation function, such as ReLU.

In PyTorch, the encoder flattens each twenty-eight by twenty-eight grayscale image, then applies the two shrinking linear layers with ReLU. The decoder applies the two growing linear layers and reshapes the result back into an image. Its output layer uses the sigmoid function rather than ReLU, so pixel values stay between zero and one. Training uses the inputs as targets and minimizes the mean squared error.

---

Reconstructions and anomaly detection. One way to check that an autoencoder is well trained is to compare inputs with their reconstructions. For the stacked autoencoder on Fashion MNIST, the reconstructions are recognizable but a bit too lossy. Training longer, making the encoder and decoder more powerful, or making the codings larger would help.

Autoencoders are commonly used for anomaly detection. When an autoencoder receives an image unlike its training images, called out of distribution, the reconstruction is terrible. Handwritten MNIST digits fed to the Fashion MNIST autoencoder come out very different from the inputs, with a very high reconstruction loss. To detect anomalies, choose a threshold: any input whose reconstruction loss exceeds it is flagged as an anomaly.

---

Visualizing Fashion MNIST. Undercomplete autoencoders usually cannot reduce data to two or three dimensions and still reconstruct it properly, so they are rarely used directly for visualization. They handle huge datasets well, though, so one strategy is to use an autoencoder to reduce the data to a moderate number of dimensions, then use another algorithm to reduce it to two for plotting. For Fashion MNIST, the stacked encoder reduces each image to thirty-two dimensions, then Scikit-Learn's t-SNE reduces those to two. The resulting scatterplot shows clusters that match the classes reasonably well. The output of t-SNE can vary greatly with a different random seed, slightly different data, or a different platform.

---

Unsupervised pretraining. For a complex supervised task with little labeled data, one solution is to reuse the lower layers of a network that performs a similar task, so the new network does not have to learn the low-level features itself. Similarly, with a large dataset that is mostly unlabeled, a stacked autoencoder can be trained on all the data, and its lower layers reused to build a network for the actual task, trained on the labeled data. With very little labeled data, the pretrained layers, at least the lower ones, may be frozen.

This situation is common: collecting millions of unlabeled images is cheap, but reliable labeling usually needs humans and is slow and costly, so a few thousand labeled instances is normal, though advanced AI systems are increasingly used to label data.

---

Tying weights. When an autoencoder is neatly symmetrical, the decoder's weights can be tied to the encoder's: each decoder layer uses the transpose of the weight matrix of the corresponding encoder layer. The first decoder layer uses the transpose of the last encoder layer's weights, and so on, while each decoder layer keeps its own bias terms. This halves the number of weights, which speeds up training and limits the risk of overfitting. On Fashion MNIST, the tied version reaches a lower reconstruction error than the untied version with about half the parameters.

---

Training one autoencoder at a time. Instead of training a whole stacked autoencoder at once, shallow autoencoders can be trained one at a time and then stacked, which is called greedy layerwise training. The first autoencoder learns to reconstruct the inputs. The whole training set is then encoded with it, producing a compressed training set, and a second autoencoder is trained on that. Finally, the encoder layers are stacked in order, followed by the decoder layers in reverse order, to form the final deep autoencoder. More autoencoders can be added the same way.

One trigger of the deep learning revolution was the 2006 discovery by Geoffrey Hinton and colleagues that deep networks can be pretrained without supervision using this greedy layerwise approach. They used restricted Boltzmann machines, and in 2007 Yoshua Bengio and colleagues showed that autoencoders work just as well. For several years this was the only efficient way to train deep networks, until later techniques made it possible to train them in one shot.

---

Convolutional autoencoders. For images other than very small ones, dense autoencoders work poorly, and convolutional networks are much better suited. In a convolutional autoencoder, the encoder is a regular CNN of convolutional and pooling layers, which reduces the height and width of the inputs while increasing the depth, meaning the number of feature maps. The decoder does the reverse, upscaling the image and reducing its depth back to the original, using transposed convolution layers, or upsampling layers combined with convolutional layers.

In the book's Fashion MNIST version, convolutional and max pooling layers shrink the image from twenty-eight by twenty-eight down to three by three, and global average pooling produces a coding of thirty-two values. The decoder expands the coding into a small three by three image with a linear layer, then three transposed convolution layers with a stride of two upscale it back to twenty-eight by twenty-eight. Autoencoders can also use other architectures, such as RNNs.

So far, the constraint forcing useful features has been a small coding layer. Other constraints allow the coding layer to be as large as the input or larger, which gives an overcomplete autoencoder.

---

Denoising autoencoders. Another way to force useful features is to add noise to the inputs and train the autoencoder to recover the original, noise-free inputs. The idea dates back to the 1980s; it appears in Yann LeCun's 1987 master's thesis. In 2008, Pascal Vincent and colleagues showed that such autoencoders can be used for feature extraction, and in 2010 they introduced stacked denoising autoencoders.

The noise can be Gaussian noise added to the inputs, or randomly switched-off inputs, as in dropout. The dropout version is simply a stacked autoencoder with a dropout layer applied to the encoder's inputs, here dropping half of them; dropout is only active during training. The coding layer does not need to compress as much, since the noise already makes the task nontrivial, so in the book's version all hidden layers, including the coding layer, have one hundred twenty-eight units. This resembles BERT's masked language modeling, which reconstructs masked inputs, although BERT is not split into an encoder and a decoder.

On noisy images with half the pixels switched off, the trained denoising autoencoder reconstructs plausible images and even guesses details missing from the input, such as the top of a shoe. Besides visualization and pretraining, denoising autoencoders are a simple, efficient way to remove noise from images.

---

Sparse autoencoders. Sparsity is another constraint that often leads to good feature extraction. Adding a suitable term to the cost function pushes the autoencoder to reduce the number of active neurons in the coding layer, so each input is represented by a few activations, and each coding neuron tends to represent a useful feature. Someone allowed only a few words per month would make them count.

A basic approach uses the sigmoid activation in the coding layer, so codings lie between zero and one, a large coding layer, for example two hundred fifty-six units, and L1 regularization on the coding layer's activations: the sum of the absolute values of the codings, weighted by a sparsity hyperparameter, is added to the loss. This pushes codings toward zero, while the reconstruction loss forces at least a few to be nonzero. Using the L1 norm rather than the L2 norm makes the network keep the most important codings and eliminate the unneeded ones, rather than shrinking all of them.

A better approach is often to measure the mean activation of each coding neuron across each training batch, and penalize the model when it differs from a target sparsity, such as ten percent. The batch must not be too small, or the mean will be inaccurate. If a neuron's mean activation is zero point three and the target is zero point one, it must be pushed to activate less. The squared error between the two could be added to the loss, but the Kullback-Leibler divergence works better, because it has much stronger gradients than the squared error.

The Kullback-Leibler divergence, or KL divergence, between two discrete probability distributions P and Q is the sum, over all outcomes, of the probability of each outcome under P times the log of the ratio of its probability under P to its probability under Q. Here the two distributions are the target probability p that a coding neuron activates and the actual probability q, its mean activation over the batch. The divergence simplifies to p times the log of p over q, plus one minus p times the log of one minus p over one minus q.

In PyTorch, the sparse autoencoder returns both the reconstructions and the codings, since the loss needs both. The loss is the mean squared error plus a weighted KL divergence between the target sparsity and the batch's mean activations, summed over the coding neurons, with a tiny smoothing value to avoid division by zero. If the weight is too high, the model sticks to the target sparsity but cannot reconstruct the inputs properly, which makes it useless. If it is too low, the model mostly ignores sparsity and learns no interesting features. After training on Fashion MNIST, the coding layer has roughly ten percent sparsity.

Sparse autoencoders often produce interpretable codings. Plotting all the images for which one particular coding is unusually high, for example above its ninetieth percentile, often shows that they have something in common, such as all being shoes.

---

Variational autoencoders. Variational autoencoders, or VAEs, were introduced in 2013 by Diederik Kingma and Max Welling and quickly became one of the most popular kinds of autoencoder. They differ from the previous ones in two ways. They are probabilistic: their outputs are partly determined by chance, even after training, whereas denoising autoencoders use randomness only during training. And, most importantly, they are generative: they can produce new instances that look as if they were sampled from the training set.

A VAE has the usual encoder and decoder, with a twist. Instead of producing a coding directly, the encoder outputs a mean coding and a standard deviation. The actual coding is sampled at random from a Gaussian distribution with that mean and standard deviation, and the decoder decodes the sampled coding normally. Even when the inputs have a very convoluted distribution, the cost function gradually pushes the codings to migrate within the coding space, also called the latent space, until they look like a cloud of points sampled from a simple Gaussian. After training, generating a new instance is easy: sample a random coding from that Gaussian distribution and decode it.

Sampling is not differentiable, so it would block backpropagation into the encoder. The solution is the reparameterization trick: sample noise from a standard Gaussian, with mean zero and standard deviation one, multiply it element-wise by the standard deviation, and add the mean. This is equivalent to sampling from the desired Gaussian, but separates the deterministic part from the random part, so gradients flow back through the mean and the standard deviation. The resulting gradients are noisy but unbiased, and the noise averages out during training.

The cost function has two parts. The first is the usual reconstruction loss, such as the mean squared error. The second is the latent loss, the KL divergence between the actual distribution of the codings and the desired Gaussian distribution. The math is more involved than for sparse autoencoders, partly because the Gaussian noise limits how much information reaches the coding layer, but it simplifies. The latent loss is minus one half times the sum, over all coding dimensions, of one, plus the log of the variance, minus the variance, minus the square of the mean. The mean and standard deviation of each coding dimension are output by the encoder.

A common tweak is to have the encoder output the log of the variance, called gamma, instead of the standard deviation. The latent loss then becomes minus one half times the sum of one, plus gamma, minus the exponential of gamma, minus the squared mean. This is more numerically stable and speeds up training.

In the book's VAE for Fashion MNIST, the encoder's last layer has twice as many outputs as the coding size of thirty-two, because it outputs both the mean and the log variance, and the output is split in two. The sampling step computes the standard deviation as the exponential of half the log variance, multiplies Gaussian noise by it, and adds the mean. The model returns the reconstructions, the means and the log variances, all of which the loss needs.

The loss computes the latent loss for each instance by summing over the coding dimensions, averages it over the batch, and divides it by seven hundred eighty-four, because the reconstruction loss is a mean over all seven hundred eighty-four pixels, and the two terms must be on the same scale. It then adds the mean squared error.

---

Generating images and semantic interpolation. To generate fashion items with the trained VAE, sample random codings from a Gaussian with mean zero and variance one and decode them. Most of the generated images look fairly convincing, if a bit fuzzy, and some are poor, which is expected after only a few minutes of training with dense layers; convolutional layers would give much better results.

VAEs also allow semantic interpolation. Interpolating two images at the pixel level just looks like the two images overlaid. Interpolating between two codings instead, and decoding each intermediate coding, gives a sequence of images that gradually turns one fashion item into another. Other VAE variants use different distributions for the latent variables.

---

Discrete variational autoencoders. A discrete VAE is like a VAE, except the codings are discrete. Each coding vector is a sequence of latent codes, also called categories, each an integer from zero up to the number of possible codes minus one. With ten possible codes and coding vectors of length six, there are one million possible coding vectors. Discrete VAEs are very useful for turning continuous inputs into tokens for transformers and other models, and they are at the core of BEiT and the first DALL-E.

The natural way to make a VAE discrete is to use a categorical distribution instead of a Gaussian. The encoder then outputs logits instead of means and variances: for each position in the coding vector, one logit per possible code. Categorical sampling is not differentiable either, so a different reparameterization trick is needed. The most popular is the Gumbel-softmax trick. The Gumbel distribution models the maximum of a set of samples. Adding Gumbel noise to the logits and taking the argmax is mathematically equivalent to categorical sampling. The argmax is not differentiable, so a softmax replaces it in the backward pass, which gives a differentiable approximation. Two independent teams proposed this almost simultaneously in 2016, one from DeepMind and Oxford University, the other from Google, Cambridge University and Stanford University.

In the book's discrete VAE, PyTorch's Gumbel-softmax function is used in hard mode, so the forward pass produces one-hot vectors of the sampled codes, while the backward pass uses the softmax approximation. The logits are divided by a temperature before the softmax. The lower the temperature, the closer the softmax output gets to one-hot vectors, which only affects the backward pass. Training typically starts at a temperature of one and lowers it gradually, down to about zero point one.

The loss is again a reconstruction loss plus a weighted KL divergence, but the target latent distribution is now a uniform categorical distribution, where each of the k possible codes has probability one over k. Since the log of one over k is minus the log of k, the divergence is the sum, over codes, of each code's predicted probability times the sum of the log of that probability and the log of k. To generate new images, sample random codes uniformly, one-hot encode them, and decode them.

Another popular discrete VAE is the vector-quantized VAE, or VQ-VAE, proposed by DeepMind researchers in 2017. Its encoder outputs a sequence of embeddings instead of logits. Each embedding is replaced by the index of the nearest embedding in a trainable embedding matrix called the codebook, which yields the integer codes. The codes are then looked up in the codebook and passed to the decoder. The nearest-neighbor lookup is not differentiable, so the backward pass treats it as the identity function and gradients pass straight through. This is the straight-through estimator, which assumes the gradients around the encoder's embeddings are similar to those around the nearest codebook embeddings. VQ-VAEs are a bit tricky to implement, but training is more stable and the codes are easier to interpret.

Discrete VAEs work well for small images but not for large ones: small-scale features look good, but there are often large-scale inconsistencies. The fix is to encode the whole training set with the trained discrete VAE, so each image becomes a sequence of integers, and train a transformer on those sequences with next token prediction. The discrete VAE learns the vocabulary, and the transformer learns the grammar. To generate an image, the transformer generates a sequence of codes and the discrete VAE's decoder turns it into an image.

This two-stage approach also makes generation controllable. If a text description is fed to the transformer during training, for example as a prefix to the codes, the transformer is conditioned on the description. After training, providing a description guides it to generate matching codes. This is exactly how the first DALL-E worked.

In practice the encoder and decoder are convolutional, so the latent representation is a grid, flattened into a sequence for the transformer. An encoder might output a thirty-two by thirty-two grid of two hundred fifty-six-dimensional embeddings. After quantization, each image becomes a thirty-two by thirty-two grid of codes, each between zero and two hundred fifty-five. To generate an image, the transformer predicts one thousand twenty-four codes, which are arranged into the grid, replaced by their codebook vectors, and decoded. Image quality can be improved further by stacking several discrete VAEs, each producing a smaller grid than the last. This is a hierarchical VAE: the encoders are stacked, then the decoders in reverse order, all trained jointly, with one reconstruction loss plus one KL divergence loss per discrete VAE.

---

Generative adversarial networks. Ian Goodfellow and colleagues proposed GANs in 2014. The idea excited researchers immediately, but it took years to overcome the difficulties of training them. The idea is to make neural networks compete in the hope that the competition pushes them to excel.

A GAN has two networks. The generator takes a random coding, usually sampled from a Gaussian distribution, and outputs data, typically an image. The coding is the latent representation of the image to generate. So the generator works like the decoder of a variational autoencoder and is used the same way to generate new images, but it is trained very differently. The discriminator takes either a fake image from the generator or a real image from the training set, and must guess whether it is fake or real.

The two networks have opposite goals, so a GAN cannot be trained like a regular network. Each training iteration has two phases. In the first phase, the discriminator is trained. A batch of real images from the training set is combined with an equal number of fake images from the generator, labeled zero for fake and one for real, and the discriminator takes one training step on this batch with the binary cross-entropy loss. Only the discriminator's weights are updated.

In the second phase, the generator is trained. The generator produces another batch of fake images, and the discriminator judges them. No real images are added, and all labels are set to one, meaning real: the generator should produce images the discriminator wrongly believes are real. The discriminator's weights are frozen during this step, so only the generator's weights change.

The generator never sees a real image, yet it learns to produce convincing fakes. All it gets is the gradients flowing back through the discriminator. The better the discriminator becomes, the more information about real images those secondhand gradients carry.

---

A simple GAN for Fashion MNIST. The generator resembles an autoencoder's decoder, turning a coding of size thirty-two into a twenty-eight by twenty-eight image through dense layers with a sigmoid output. The discriminator is a regular binary classifier, a dense network ending with a single sigmoid unit.

The custom training function uses the binary cross-entropy loss and a separate optimizer for each network. In phase one, real images are scored against targets of one, and fake images, detached from the computation graph so the generator is unaffected, are scored against targets of zero; the two losses are added and the discriminator takes a gradient step. In phase two, new fakes are generated without detaching them, the discriminator's parameters are frozen by turning off their gradient tracking, the fakes are scored against targets of one, the generator takes a gradient step, and the discriminator is unfrozen.

After training, new images are generated by feeding random Gaussian codings to the generator. After the first epoch, the generated images already look like very noisy Fashion MNIST items. Unfortunately, they never get much better, and some epochs even seem to forget what was learned.

---

The difficulties of training GANs. The generator and discriminator play a zero-sum game, constantly trying to outsmart each other. The game may reach what game theorists call a Nash equilibrium, named after the mathematician John Nash: a state where no player would be better off changing strategy, assuming the others keep theirs. Everyone driving on the left side of the road is a Nash equilibrium, since no single driver gains by switching sides, and everyone driving on the right is another.

The authors of the GAN paper showed that a GAN can reach only one Nash equilibrium: the generator produces perfectly realistic images, and the discriminator is reduced to guessing, fifty percent real and fifty percent fake. That seems to promise that long enough training yields a perfect generator, but nothing guarantees the equilibrium will ever be reached.

The biggest difficulty is mode collapse, when the generator's outputs gradually become less diverse. Suppose the generator gets better at convincing shoes than at any other class. Shoes fool the discriminator a little more, which encourages the generator to make more shoes, until it forgets how to make anything else. Meanwhile the discriminator sees only fake shoes, so it forgets how to judge fakes of other classes. When it finally learns to spot the fake shoes, the generator moves to another class, say shirts, forgetting shoes, and the discriminator follows. The GAN may cycle through a few classes without ever becoming very good at any.

Also, because the two networks push against each other, their parameters can oscillate and become unstable. Training may start well, then suddenly diverge for no apparent reason. And since many factors affect these dynamics, GANs are very sensitive to hyperparameters, which can take a lot of effort to tune.

Many papers since 2014 have proposed new cost functions, though a 2018 paper by Google researchers questioned their efficiency, and techniques to stabilize training or avoid mode collapse. Experience replay stores the images produced by the generator at each iteration in a replay buffer, gradually dropping older ones, and trains the discriminator on real images plus fakes drawn from the buffer, rather than only the current generator's fakes. This reduces the chance that the discriminator overfits the latest generator's outputs. Mini-batch discrimination measures how similar images are across a batch and gives this statistic to the discriminator, so it can easily reject a whole batch of fakes that lack diversity. This pushes the generator toward variety and reduces the risk of mode collapse.

GAN research made steady progress: deep convolutional GANs, or DCGANs, built on convolutional layers; progressively growing GANs that produced high-resolution images; and StyleGANs that gave fine-grained control over generation. GANs seemed to have a bright future, but when diffusion models began producing amazing images with much more stable training and more diverse outputs, GANs were quickly sidelined.

---

Diffusion models. The ideas behind diffusion models are old, but they were formalized in their modern form in a 2015 paper by Jascha Sohl-Dickstein and colleagues from Stanford University and UC Berkeley. They applied tools from statistical mechanics to model a diffusion process, like a drop of milk spreading through a cup of tea, and trained a model to learn the reverse process: starting from the fully mixed state and gradually unmixing the milk from the tea. Results were promising, but GANs produced more convincing images, much faster, so diffusion models got little attention.

In 2020, Jonathan Ho and colleagues, also from UC Berkeley, built a diffusion model that generated highly realistic images, called a denoising diffusion probabilistic model, or DDPM. In 2021, OpenAI researchers Alex Nichol and Prafulla Dhariwal analyzed the DDPM architecture and proposed improvements that let DDPMs finally beat GANs: they are much easier to train, and their images are more diverse and of even higher quality. Their main drawback is that generating an image takes a very long time compared with GANs or VAEs.

---

The forward process. Start with a picture of a cat. At each time step, add a little Gaussian noise to the image, with mean zero and a small variance called beta for that step. The noise is independent for each pixel, with the same mean and variance, which is called isotropic. After many steps, the cat is completely hidden by noise. The original DDPM paper used one thousand time steps, with a variance schedule that made the cat signal fade linearly. The improved DDPM paper used four thousand steps, with a schedule that changes more slowly at the beginning and at the end. Gradually drowning the image in noise is called the forward process.

As noise accumulates, the distribution of pixel values becomes more and more Gaussian. At each step, the pixel values are also scaled by the square root of one minus beta, a factor slightly below one. This makes the mean drift toward zero, like repeatedly multiplying a number by zero point nine nine. It also makes the variance converge to one: the old variance is multiplied by one minus beta, the square of the scaling factor, and the added noise contributes beta, since variances add when independent Gaussians are summed. The variance therefore settles at one, the only value that stays unchanged when multiplied by one minus beta and then increased by beta.

There is a shortcut. Since a sum of independent Gaussians is also Gaussian, all the noise up to step t can be added in one shot. Define alpha at each step as one minus beta, and alpha bar at step t as the product of all the alphas up to t. Then the image at step t, given the original image, is Gaussian, with a mean equal to the original image times the square root of alpha bar, and a variance of one minus alpha bar. This shortcut is much faster and is what the implementation uses.

The purpose is not to drown cats in noise, but to train a model to perform the reverse process, going from one step back to the previous one. Applied repeatedly, it removes noise a little at a time until none is left. It is not a simple noise filter that relies on neighboring pixels: the removed noise is replaced with realistic content, depending on the training data. A model trained on many cat images, given an image of pure Gaussian noise, gradually makes a brand new cat appear.

---

The variance schedule. At each step the part of the image's variance that comes from the original cat shrinks by a factor of alpha, so after t steps it has shrunk by alpha bar. This cat signal factor, alpha bar, is what gets scheduled, shrinking gradually from one to zero between the first and last time step. The improved DDPM paper uses a cosine schedule. A function f of t is defined as the squared cosine of an angle that grows from nearly zero to a right angle as t goes from zero to the last step; specifically, t over the number of steps, plus a small offset s, all divided by one plus s, times pi over two. Alpha bar at step t is f of t divided by f of zero. Beta at step t is one minus the ratio of alpha bar at t to alpha bar at the previous step. The offset s, set to zero point zero zero eight, stops beta from being too small near the start, and beta is clipped to at most zero point nine nine nine to avoid instabilities near the end.

---

Training a diffusion model. Training needs noisy images from random time steps. A forward diffusion function takes an original image and a time step, samples Gaussian noise of the same shape, and returns the noisy image, computed as the square root of alpha bar times the original image plus the square root of one minus alpha bar times the noise, along with the unscaled noise itself, which serves as the target. A dataset wrapper takes Fashion MNIST, rescales pixel values to range from minus one to one, which is optional but usually works better, picks a random time step, applies forward diffusion, and returns the noisy image and time step as the input and the unscaled noise as the target.

The model predicts the noise rather than the original image for two reasons. Empirically, the authors found that predicting the noise gave more stable training and better results. And because the noise is Gaussian, the math simplifies: the KL divergence between two Gaussians is proportional to the squared distance between their means, so a simple, fast and stable loss like the mean squared error can be used.

The diffusion model can be any model that takes a noisy image and its time step and outputs an image of the same shape. The DDPM authors used a modified U-Net, which resembles the fully convolutional network used for semantic segmentation. A U-Net is a CNN that gradually downsamples its input, then gradually upsamples it again, with skip connections from each downsampling level to the matching upsampling level. The time step is encoded with a fixed sinusoidal encoding, the same technique as the original Transformer's positional encodings, passed through linear layers, and fed in at every level of the U-Net. Multi-head attention layers are also used at several levels. The details matter little; many architectures work.

The authors found that the mean absolute error loss worked better than the mean squared error, and the Huber loss is another option. The book trains with the Huber loss and the NAdam optimizer.

---

Generating images with DDPM. To generate an image, sample pure noise from a Gaussian with mean zero and variance one, then apply the reverse step repeatedly, four thousand times, until reaching the final image. Each reverse step computes the previous image as follows: take the current image, subtract the model's predicted noise scaled by beta divided by the square root of one minus alpha bar, divide the result by the square root of alpha, then add fresh Gaussian noise scaled by the square root of beta. The predicted noise depends on the current image, the time step, and the model's parameters. The fresh noise makes the reverse process stochastic, so repeated runs give different images.

---

DDIM. Four thousand iterations per image is too slow. A few months after the DDPM paper, Stanford researchers proposed the denoising diffusion implicit model, or DDIM, which can jump down any number of time steps at a time, for example fifty. Training is exactly the same as for DDPM, so a trained DDPM model can be reused.

At each jump from the current step to an earlier one, DDIM first estimates the clean image: the current image minus the square root of one minus alpha bar times the predicted noise, all divided by the square root of alpha bar. The new image is then the square root of the earlier step's alpha bar times that estimate, plus the predicted noise scaled by a coefficient, plus fresh noise scaled by a value sigma. The noise coefficient is the square root of one minus the earlier alpha bar minus sigma squared. Sigma squared equals a hyperparameter eta, times the ratio of one minus the earlier alpha bar to one minus the current alpha bar, times beta. Eta controls the randomness, from zero, fully deterministic, to one, which behaves like DDPM.

The book's generation function applies this update over evenly spaced time steps with an eta of zero point eight five, then rescales the pixels back to the range zero to one. With five hundred steps, generation takes only a few seconds. The images are not very impressive after a few minutes of training on Fashion MNIST; a larger dataset and hours of training do much better.

---

Latent diffusion and Stable Diffusion. A December 2021 paper by Robin Rombach and colleagues introduced latent diffusion models, where diffusion happens in a latent space rather than in pixel space. A powerful autoencoder compresses each training image into a much smaller latent representation, diffusion takes place there, and the autoencoder decompresses the final latent representation into the output image. This considerably speeds up generation, dramatically reduces training time and cost, and produces outstanding image quality.

The researchers also adapted conditioning techniques to guide diffusion with text prompts, images or other inputs, which makes it possible to produce almost any desired image quickly. Conditioning on an input image enables outpainting, which extends an image beyond its borders, and inpainting, which fills holes in an image.

In August 2022, a powerful pretrained latent diffusion model called Stable Diffusion was open sourced by a collaboration between LMU Munich and several companies, including Stability AI and Runway, with support from EleutherAI and LAION. Anyone can now generate remarkable images in seconds, for free, even on a regular laptop. With the Hugging Face Diffusers library, the book loads the turbo variant of Stable Diffusion in half precision through an automatic text-to-image pipeline, and generates a closeup photo of an orangutan reading a book in a single inference step, with guidance turned off.

---

Summary.

Autoencoders learn compact latent representations by learning to copy their inputs under constraints: a small coding layer, noise on the inputs, or sparsity. A linear autoencoder with the mean squared error performs principal component analysis. Stacked autoencoders are symmetrical sandwiches; their weights can be tied, and they can be trained one layer at a time. Convolutional autoencoders suit images. Autoencoders are used for dimensionality reduction and visualization, anomaly detection through high reconstruction error, unsupervised pretraining, and denoising.

Variational autoencoders are probabilistic and generative. The encoder outputs a mean and a variance, a coding is sampled with the reparameterization trick, and a latent KL divergence loss pushes the codings toward a simple Gaussian, so new instances come from decoding random Gaussian codings. Discrete VAEs, using the Gumbel-softmax trick or vector quantization, turn images into sequences of codes that a transformer can model, which is how the first DALL-E generated images from text.

GANs pit a generator against a discriminator, alternating between training each. They can produce sharp images quickly, but training is hard: mode collapse, oscillation and sensitivity to hyperparameters. Experience replay and mini-batch discrimination help.

Diffusion models learn to reverse a process that gradually adds Gaussian noise. A DDPM is trained to predict the noise added at a random step, using a closed-form shortcut for the forward process, and generates images by denoising pure noise step by step. Generation is slow, but DDIM skips steps with the same trained model. Latent diffusion runs the process in a compressed latent space with conditioning on text or images, which led to Stable Diffusion. Diffusion models are easier to train than GANs and produce more diverse, higher-quality images, at the cost of slower generation.
