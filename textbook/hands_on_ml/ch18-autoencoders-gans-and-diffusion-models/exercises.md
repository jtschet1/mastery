# Chapter 18 — Autoencoders, GANs, and Diffusion Models

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. The author
hasn't published solutions for this chapter yet (the exercise section of
`handson-mlp-main/18_autoencoders_gans_and_diffusion_models.ipynb` says it's
a work in progress), so the questions below have no answers. build_cards.py
skips a card until it has one, so write yours under each heading as you
read.

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

## 2. Suppose you want to train a classifier, and you have plenty of unlabeled training data but only a few thousand labeled instances. How can autoencoders help? How would you proceed?

## 3. If an autoencoder perfectly reconstructs the inputs, is it necessarily a good autoencoder? How can you evaluate the performance of an autoencoder?

## 4. What are undercomplete and overcomplete autoencoders? What is the main risk of an excessively undercomplete autoencoder? What about the main risk of an overcomplete autoencoder?

## 5. How do you tie weights in a stacked autoencoder? What is the point of doing so?

## 6. What is a generative model? Can you name a type of generative autoencoder?

## 7. What is a GAN? Can you name a few tasks where GANs can shine?

## 8. What are the main difficulties when training GANs?

## 9. What are diffusion models good at? What is their main limitation?
