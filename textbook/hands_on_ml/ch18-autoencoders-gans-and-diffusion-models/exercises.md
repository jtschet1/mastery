# Chapter 18 — Autoencoders, GANs, and Diffusion Models

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. The author
hasn't published solutions for this chapter yet (the exercise section of
`handson-mlp-main/18_autoencoders_gans_and_diffusion_models.ipynb` says it's
a work in progress), so the answers below were written by Claude from the
chapter text, not by the author. If the author publishes solutions later,
swap theirs in.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text.

- 10. Pretrain an image classifier with a denoising autoencoder (MNIST or
  CIFAR10): check the reconstructions and what activates each coding neuron,
  then reuse the encoder's lower layers in a classifier trained on only 500
  labeled images, with and without pretraining.
- 11. Train a variational autoencoder on an image dataset of your choice and
  generate new images.
- 12. Train a DCGAN on an image dataset of your choice and generate images;
  then add experience replay.
- 13. Train a diffusion model (e.g., on torchvision.datasets.Flowers102),
  then add the image class as an extra input so you can control what it
  generates.

## 1. What are the main tasks that autoencoders are used for?

Autoencoders are mainly used for the following tasks:
- Dimensionality reduction: an undercomplete autoencoder compresses its inputs into compact codings. For visualization, a common strategy is to compress a large dataset this way first, then use another algorithm such as t-SNE to get down to 2D.
- Feature extraction: the codings act as learned feature detectors (sparse autoencoders often produce fairly interpretable ones).
- Unsupervised pretraining: train an autoencoder on plenty of unlabeled data, then reuse its lower layers in a model trained on the few labeled instances.
- Anomaly detection: an input unlike the training data (out of distribution) is poorly reconstructed, so a reconstruction loss above a chosen threshold flags an anomaly.
- Denoising: a denoising autoencoder can remove noise from images.
- Generating new data: generative autoencoders such as VAEs produce new instances that look like the training data (and allow semantic interpolation between instances).
- Compact inputs for other models: discrete VAEs turn images into sequences of codes that a transformer can model (as in DALL·E), and latent diffusion models run the diffusion process in an autoencoder's small latent space.

## 2. Suppose you want to train a classifier, and you have plenty of unlabeled training data but only a few thousand labeled instances. How can autoencoders help? How would you proceed?

Autoencoders let you learn from all the unlabeled data: an autoencoder trained on it learns useful feature detectors, which the classifier can then reuse instead of having to learn every low-level feature from a few thousand labeled instances. This is called unsupervised pretraining. Here's how to proceed:
1. Train an autoencoder (e.g., a stacked, convolutional, or denoising autoencoder) on the whole training set, labeled and unlabeled instances alike, since it doesn't need labels. Check that it reconstructs held-out images reasonably well.
2. Build the classifier by reusing the autoencoder's encoder (its lower layers) and adding new layers on top, ending with an output layer suited to the classification task.
3. Train the classifier on the labeled instances. Since the new top layers start out random, freeze the reused layers at first (with very little labeled data, you may want to keep at least the lowest ones frozen), then consider unfreezing some of them and fine-tuning with a lower learning rate.
4. Compare the result with the same classifier trained from scratch on the labeled data only, to check that the pretraining actually helps.

## 3. If an autoencoder perfectly reconstructs the inputs, is it necessarily a good autoencoder? How can you evaluate the performance of an autoencoder?

No. A perfect reconstruction may just mean that the autoencoder found a way to copy its inputs without learning anything useful. An overcomplete autoencoder with no other constraint can simply learn the identity function. Even with a tiny coding layer, a powerful enough encoder could learn to map each training instance to an arbitrary number, with the decoder learning the reverse mapping: the training data would be reconstructed perfectly, but the codings would be meaningless and the model would likely generalize poorly to new instances. Very poor reconstructions, on the other hand, clearly signal a bad autoencoder.

To evaluate an autoencoder:
- Measure its reconstruction loss (e.g., the MSE between the inputs and the outputs) on a validation or test set, and visually compare some inputs with their reconstructions. A high loss shows that the autoencoder is bad, but a low loss doesn't prove that it's good.
- Evaluate it on the task it was built for. For example, if it's used for unsupervised pretraining, measure the performance of the classifier that reuses its encoder; if it's used for anomaly detection, check how well it flags anomalies; and if it's a generative autoencoder, look at the quality and diversity of the instances it generates.

## 4. What are undercomplete and overcomplete autoencoders? What is the main risk of an excessively undercomplete autoencoder? What about the main risk of an overcomplete autoencoder?

An undercomplete autoencoder has codings of lower dimensionality than its inputs (e.g., 32 codings for 784-pixel images). Since it can't just copy its inputs to its codings, it must compress the data, which pushes it to keep the most important features and discard the rest. An overcomplete autoencoder has a coding layer as large as its inputs, or even larger.

The main risk of an excessively undercomplete autoencoder is that its codings are too small to hold the information needed to reconstruct the inputs: the reconstructions become too lossy, and useful information is thrown away. That's why autoencoders are rarely used on their own to reduce data all the way down to 2 or 3 dimensions.

The main risk of an overcomplete autoencoder is that it just learns to copy its inputs to its outputs (the identity function) without learning any useful features. It therefore needs some other constraint to force it to learn useful representations, such as adding noise to the inputs (denoising autoencoder), penalizing active codings (sparse autoencoder), or the latent loss and sampling noise of a variational autoencoder.

## 5. How do you tie weights in a stacked autoencoder? What is the point of doing so?

You tie weights by making each decoder layer reuse the transposed weight matrix of the corresponding encoder layer instead of having its own weights. This requires a stacked autoencoder that's symmetrical around its coding layer. If it has n layers (not counting the input layer) and Wₗ is the weight matrix of layer l, with layer n/2 being the coding layer, then the decoder layers use Wₗ = Wₙ₋ₗ₊₁ᵀ for l = n/2 + 1, …, n. Only the weights are shared: each decoder layer keeps its own bias vector. In PyTorch, you can create nn.Linear layers for the encoder only, plus nn.Parameter bias vectors for the decoder, and implement each decoder layer with F.linear(), passing it the transposed weights of the mirror encoder layer (e.g., self.enc2.weight.t()) and the decoder layer's own bias.

The point is to roughly halve the number of weights in the model, which makes training faster and reduces the risk of overfitting. On Fashion MNIST, a tied autoencoder even reached a lower reconstruction error than its untied counterpart, with about half the parameters.

## 6. What is a generative model? Can you name a type of generative autoencoder?

A generative model is one that can produce new, random instances that resemble the training data, as if they had been drawn from the same distribution. For example, a generative model trained on pictures of faces can produce new, realistic faces of people who don't exist. Some can also be guided, for instance by conditioning them on a text description of the desired output.

The main type of generative autoencoder is the variational autoencoder (VAE). Instead of producing a coding directly, its encoder outputs a mean coding and a standard deviation (usually as the log of the variance), and the actual coding is sampled from that Gaussian distribution. Its cost function adds a latent loss to the reconstruction loss, pushing the codings to look like samples from a standard Gaussian distribution. So after training, you can generate a new instance simply by sampling a random coding from that Gaussian distribution and decoding it. Variants include discrete VAEs (whose codes follow a categorical distribution) and hierarchical VAEs. GANs and diffusion models are generative models too, but they aren't autoencoders.

## 7. What is a GAN? Can you name a few tasks where GANs can shine?

A generative adversarial network (GAN) is composed of two neural networks with opposite goals. The generator turns a random coding (usually drawn from a Gaussian distribution) into data, usually an image, much like a VAE's decoder. The discriminator is a binary classifier: it's shown either a real image from the training set or a fake one from the generator, and it must tell which is which. Each training iteration has two phases: first, the discriminator is trained for one step on a batch of real images (labeled 1) and fake images (labeled 0); then the generator is trained for one step, with the discriminator frozen, to make the discriminator classify its fake images as real. Interestingly, the generator never gets to see a real image: its only training signal is the gradient that flows back to it through the discriminator.

GANs can shine at increasing image resolution (super-resolution), colorizing images, advanced image editing such as replacing photobombers with a realistic background, turning rough sketches into photorealistic images, predicting the upcoming frames of a video, generating extra training data to augment a dataset, generating other kinds of data such as text, audio, or time series, and exposing weaknesses in other models so they can be strengthened. Diffusion models have largely replaced them for high-quality image generation, but GANs remain useful when generation must be very fast.

## 8. What are the main difficulties when training GANs?

The generator and the discriminator play a zero-sum game against each other, which causes several difficulties:
- No guarantee of convergence: a GAN's only Nash equilibrium is reached when the generator's images are perfectly realistic, leaving the discriminator no better than a coin toss (50% real, 50% fake), yet there's no guarantee that training will ever get there.
- Mode collapse, the biggest difficulty: the generator's outputs gradually lose diversity. If it gets slightly better at producing, say, shoes, it produces more and more shoes and forgets everything else, while the discriminator, seeing only fake shoes, forgets how to spot other fakes. When the discriminator catches up, the generator moves to another class, and the GAN may keep cycling across a few classes without mastering any.
- Instability: as the two networks push against each other, their parameters can keep oscillating; training may go well for a while, then blow up or seem to forget what it had learned.
- Hyperparameter sensitivity: the dynamics depend on many factors, so getting the hyperparameters right can take a lot of tuning.

## 9. What are diffusion models good at? What is their main limitation?

A diffusion model is trained to denoise images a little at a time, so you can generate a brand-new image by starting from pure random noise and repeatedly denoising it. Diffusion models are good at generating high-quality, highly realistic images that are also more diverse than those produced by GANs, and they are much easier and more stable to train than GANs, since training is a simple regression task (predicting the noise that was added to an image) rather than an adversarial game. They are also easy to guide by conditioning them on text prompts or input images, which enables text-to-image generation, inpainting (filling holes in an image), and outpainting (extending an image beyond its borders). Pretrained latent diffusion models such as Stable Diffusion let anyone generate impressive images in seconds.

Their main limitation is slow generation: the model must be run once per denoising step, so generating a single image with a DDPM takes thousands of steps (e.g., 1,000 or 4,000), whereas a GAN or a VAE generates an image in a single forward pass. DDIM sampling (which skips many steps) and latent diffusion (which runs the process in a much smaller latent space) reduce this cost considerably, but GANs are still preferred when generation must be very fast.
