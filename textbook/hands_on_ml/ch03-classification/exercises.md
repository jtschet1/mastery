# Chapter 3 — Classification

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). This chapter's exercises are all hands-on coding, so it
has no flashcards: do them in a notebook. The author's solutions are at the
end of `handson-mlp-main/03_classification.ipynb`.

### Hands-on exercises

Summarized here; see the book for the full text.

- 1. Build an MNIST classifier with over 97% test accuracy (hint:
  grid-search KNeighborsClassifier's weights and n_neighbors).
- 2. Data augmentation: shift every MNIST training image one pixel in each
  direction, add the copies to the training set, retrain your best model,
  and compare.
- 3. Tackle the Titanic dataset: predict Survived from the other columns.
- 4. Build a spam classifier from Apache SpamAssassin's data: turn emails
  into word-presence or word-count vectors, then try several classifiers for
  high precision and recall.
