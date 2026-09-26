# Chapter 1 — The Machine Learning Landscape

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. What is self-supervised learning, and how is a self-supervised model usually put to work on the task you actually care about?

Self-supervised learning generates labels from unlabeled data itself, so any supervised algorithm can train on it. For example, mask a small patch of each image and train a model to restore it: the masked image is the input, the original is the label. Language models are pretrained similarly, by predicting hidden words in huge text corpora.

The pretrained model is rarely the end goal. To fill in blanks well it must learn what the data looks like (restoring a cat's masked face means knowing cats from dogs), so you adapt it to the task you care about, say classifying pet species, and fine-tune it on a labeled dataset, which can be much smaller. Reusing knowledge from one task on another is called transfer learning.

It's best treated as its own category: the data is unlabeled, yet training uses generated labels and targets supervised-style tasks such as classification, not clustering.
