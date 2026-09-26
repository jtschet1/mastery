# Chapter 18 — Autoencoders, GANs, and Diffusion Models

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's end-of-chapter questions are in
exercises.md, with answers written by Claude, since the author hasn't
published any; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. How does a denoising autoencoder work, and how do you implement the dropout version in PyTorch?

You corrupt the inputs during training and train the autoencoder to reconstruct the original, clean inputs. Since it can't simply copy what it sees, it has to learn the structure of the data, so the coding layer doesn't need to compress as much (it can even be as large as the input). The corruption is either Gaussian noise added to the inputs or randomly switched-off inputs, as in dropout.

The dropout version is a regular stacked autoencoder whose encoder starts with a Dropout layer, and the targets are still the clean images:

```python
encoder = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.5),                     # zeroes each pixel with probability 0.5
    nn.Linear(28 * 28, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU())
```

Dropout is only active in training mode, so after model.eval() the inputs pass through untouched. Besides feature learning and pretraining, a trained denoising autoencoder can be used directly to clean up noisy images.

## 2. What is a sparse autoencoder, and how does the KL-divergence sparsity loss work?

A sparse autoencoder adds a loss term that keeps most coding units inactive for any given input, so each input is represented by a few active units, which tend to become interpretable features. The coding layer is usually large (e.g., 256 units), with a sigmoid activation so codings lie in [0, 1].

- Simple option: add the ℓ1 norm of the codings to the loss, times a weight. Unlike ℓ2, it drives unneeded codings to 0 rather than shrinking them all.
- Often better: compute each coding unit's mean activation q over the batch (so batches can't be too small) and add D_KL(p ‖ q) = p·log(p/q) + (1 − p)·log((1 − p)/(1 − q)), summed over the units, where p is the target sparsity (e.g., 0.1). Its gradients are much stronger than those of the squared error (p − q)².

The weight is a trade-off: too high a weight hurts reconstruction, while too low a weight lets the model ignore sparsity.

## 3. How does a variational autoencoder's encoder produce codings, and what is the reparameterization trick?

The encoder doesn't output a coding directly. It outputs the parameters of a Gaussian, a mean μ and (usually) the log-variance γ = log σ², and the coding is sampled from N(μ, σ²). So a VAE's outputs are partly random even after training.

Sampling isn't differentiable, so gradients couldn't reach the encoder. The reparameterization trick samples ε from N(0, I) instead and computes z = μ + σ ⊙ ε: the randomness is isolated in ε, and gradients flow back through μ and σ. In PyTorch, the encoder's last layer outputs 2 × codings_dim values, which are split in two:

```python
def encode(self, X):
    return self.encoder(X).chunk(2, dim=-1)   # (mean, logvar)

def sample_codings(self, mean, logvar):
    std = torch.exp(0.5 * logvar)             # σ = exp(γ / 2)
    return mean + torch.randn_like(std) * std
```

Once the VAE is trained, you generate new instances by sampling z from N(0, I) and passing it to the decoder.

## 4. Write out a VAE's latent loss in terms of μ and γ = log σ², and explain what it does and how it's combined with the reconstruction loss.

ℒ = −½ ∑ᵢ [1 + γᵢ − exp(γᵢ) − μᵢ²], summed over the n coding dimensions, where μᵢ and γᵢ are the encoder's outputs for dimension i.

It's the KL divergence between the encoder's Gaussian and the standard normal distribution. Rewritten as ½ ∑ᵢ [μᵢ² + (exp(γᵢ) − γᵢ − 1)], the first part pulls the means toward 0, and the second is 0 only when γᵢ = 0, i.e., σᵢ = 1. So the codings come to look like samples from N(0, I), which is what lets you generate new data by sampling from N(0, I) and decoding.

The total loss is the reconstruction loss (e.g., MSE) plus a weighted latent loss. Mind the scales: if the MSE is averaged over the pixels (784 for Fashion MNIST) while the latent loss is summed over coding dimensions, divide the latent loss by the number of pixels so the two terms are comparable.

## 5. How do discrete VAEs get gradients through the choice of discrete codes? Contrast Gumbel-softmax with VQ-VAE.

A discrete VAE encodes each input as d integer codes, each between 0 and k − 1, so it can act as a tokenizer: a transformer can learn to generate sequences of codes (as in the first DALL·E), and the decoder turns them into images. Picking a code isn't differentiable, so:

- Gumbel-softmax: the encoder outputs logits of shape [d, k]. Adding Gumbel noise to the logits and taking the argmax is equivalent to categorical sampling, and the backward pass replaces the argmax with a softmax. F.gumbel_softmax(logits, tau=temperature, hard=True) does exactly this: one-hot codes forward, softmax gradients backward. The temperature is usually annealed from 1 down to about 0.1.
- VQ-VAE: the encoder outputs d embeddings of size e, and each is replaced by the nearest of the k vectors in a trainable codebook of shape [k, e]. The backward pass treats this lookup as the identity function (the straight-through estimator). Training tends to be more stable.

## 6. Walk through one GAN training iteration in PyTorch.

Each iteration has two phases, each with its own optimizer. Here bce is nn.BCELoss() (the discriminator ends with a sigmoid), and ones and zeros are label tensors of shape [batch_size, 1]:

```python
# Phase 1: train the discriminator (real images → 1, fakes → 0)
z = torch.randn(batch_size, codings_dim, device=device)
fake = generator(z).detach()                # no gradients for the generator
d_loss = bce(discriminator(real), ones) + bce(discriminator(fake), zeros)
d_opt.zero_grad(); d_loss.backward(); d_opt.step()
# Phase 2: train the generator to make the discriminator say "real"
discriminator.requires_grad_(False)         # freeze the discriminator
z = torch.randn(batch_size, codings_dim, device=device)
g_loss = bce(discriminator(generator(z)), ones)
g_opt.zero_grad(); g_loss.backward(); g_opt.step()
discriminator.requires_grad_(True)
```

In phase 2 the fakes are deliberately labeled "real", and only the generator's weights change. The generator never sees a real image: all it learns from is the gradient flowing back through the discriminator.

## 7. How do experience replay and mini-batch discrimination help prevent mode collapse in GANs?

Mode collapse is when the generator's outputs gradually lose diversity. Both techniques change what the discriminator gets to see:

- Experience replay: store the images the generator produces over time in a replay buffer (gradually dropping the oldest), and train the discriminator on real images plus fakes sampled from that buffer, rather than only on the current generator's output. The discriminator is then less likely to overfit the latest generator's outputs.
- Mini-batch discrimination: measure how similar the images in a batch are to each other and give that statistic to the discriminator. A batch of fakes lacking diversity becomes easy to reject, so the generator is pushed to produce varied images.

## 8. Explain the DDPM forward (noising) process and its closed-form shortcut.

Starting from a training image x₀, each step t scales the image by √(1 − βₜ) and adds Gaussian noise of mean 0 and variance βₜ, independently for every pixel (isotropic noise):

q(xₜ | xₜ₋₁) = N(√(1 − βₜ)·xₜ₋₁, βₜ·I)

The variance schedule βₜ sets how fast the image fades. The scaling drives the mean to 0, and since each step maps the variance v to (1 − βₜ)·v + βₜ, the variance converges to 1, so after the last step T (typically thousands), x_T is essentially pure N(0, I) noise.

A sum of independent Gaussians is Gaussian, so you can jump straight to any step. With αₜ = 1 − βₜ and ᾱₜ = α₁·α₂·…·αₜ:

q(xₜ | x₀) = N(√ᾱₜ·x₀, (1 − ᾱₜ)·I), i.e., xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε with ε ~ N(0, I)

ᾱₜ is the fraction of the original image's variance that remains; a cosine schedule makes it decay smoothly from 1 to 0.

## 9. How is a DDPM trained, what exactly does the model predict, and why?

Training pairs are built on the fly: take an image x₀ (pixels rescaled to [−1, 1]), pick a random step t between 1 and T, sample ε ~ N(0, I), and noise the image in one shot: xₜ = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε, where ᾱₜ is the fraction of the image's variance left at step t. The input is the pair (xₜ, t), and the target is the unscaled noise ε.

The model ε_θ(xₜ, t) outputs a tensor shaped like the image. DDPM used a U-Net (a CNN that downsamples then upsamples, with skip connections between matching levels) that also gets a sinusoidal encoding of t at every level. The loss compares predicted and actual noise: MSE, or MAE, which worked better in practice (Huber works too).

Predicting the noise rather than the clean image gave more stable training and better results, and since the noise is Gaussian, the KL-based objective reduces to a squared error.

## 10. How do you generate images with a trained DDPM, and how does DDIM make it faster?

Start from pure noise x_T ~ N(0, I) and apply the reverse step for t = T, …, 1:

xₜ₋₁ = (1/√αₜ)·(xₜ − (βₜ/√(1 − ᾱₜ))·ε_θ(xₜ, t)) + √βₜ·z, with z ~ N(0, I)

Here ε_θ is the trained noise predictor, βₜ the noise variance of step t, αₜ = 1 − βₜ, and ᾱₜ = α₁·α₂·…·αₜ. Each step removes the scaled predicted noise, rescales, and adds a little fresh noise, so runs differ. At one model call per step, thousands in total, it's slow.

DDIM reuses the same trained model but can jump from step t to any earlier step p (e.g., 50 steps at a time). It first estimates the clean image, x̂₀ = (xₜ − √(1 − ᾱₜ)·ε_θ(xₜ, t))/√ᾱₜ, then re-noises it to level p: xₚ = √ᾱₚ·x̂₀ + √(1 − ᾱₚ − σₜ²)·ε_θ(xₜ, t) + σₜ·z. A hyperparameter η from 0 to 1 scales the fresh noise σₜ: η = 0 is fully deterministic, η = 1 behaves like DDPM.

## 11. What is a latent diffusion model, and how do you generate an image with a pretrained one using Hugging Face Diffusers?

A latent diffusion model runs the diffusion process in the compact latent space of a powerful autoencoder instead of in pixel space: the encoder compresses the training images, the model learns to denoise latents, and at generation time the decoder turns the final denoised latent into an image. Working on much smaller tensors makes training far cheaper and generation much faster, with outstanding quality. The process can also be conditioned on a text prompt or an image, enabling text-to-image generation, inpainting, and outpainting. Stable Diffusion is an open-source pretrained one:

```python
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo",
                                                 variant="fp16")
pipe.to(device)
image = pipe(prompt="an oil painting of a lighthouse at dawn",
             num_inference_steps=1, guidance_scale=0.0).images[0]
```

variant="fp16" fetches the half-precision weights, and .images is a list of PIL images. SD-Turbo is distilled to work in a single denoising step, and guidance_scale=0.0 disables classifier-free guidance, which it doesn't use.
