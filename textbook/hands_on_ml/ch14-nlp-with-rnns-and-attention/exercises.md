# Chapter 14 — Natural Language Processing with RNNs and Attention

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. Answers below
are the author's, from the exercise solutions at the end of
`handson-mlp-main/14_nlp_with_rnns_and_attention.ipynb`, converted to plain
text for the flashcard app.

Text in [brackets] is added, not the author's: it replaces details left over
from the book's TensorFlow edition with what this edition uses.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text. Solutions are in the same
notebook unless noted.

- 7. Generate strings from an embedded Reber grammar (about half valid, half
  not) and train an RNN to tell whether a string follows the grammar. No
  author solution yet.
- 8. Train an encoder-decoder model that converts dates from one format to
  another (e.g., "April 22, 2019" to "2019-04-22"). No author solution yet.

## 1. What are the pros and cons of using a stateful RNN versus a stateless RNN?

Stateless RNNs can only capture patterns whose length is less than, or equal to, the size of the windows the RNN is trained on. Conversely, stateful RNNs can capture longer-term patterns. However, implementing a stateful RNN is much harder—especially preparing the dataset properly. Moreover, stateful RNNs do not always work better, in part because consecutive batches are not independent and identically distributed (IID). Gradient Descent is not fond of non-IID datasets.

## 2. Why do people use encoder-decoder RNNs rather than plain sequence-to-sequence RNNs for automatic translation?

In general, if you translate a sentence one word at a time, the result will be terrible. For example, the French sentence "Je vous en prie" means "You are welcome," but if you translate it one word at a time, you get "I you in pray." Huh? It is much better to read the whole sentence first and then translate it. A plain sequence-to-sequence RNN would start translating a sentence immediately after reading the first word, while an Encoder–Decoder RNN will first read the whole sentence and then translate it. That said, one could imagine a plain sequence-to-sequence RNN that would output silence whenever it is unsure about what to say next (just like human translators do when they must translate a live broadcast).

## 3. How can you deal with variable-length input sequences? What about variable-length output sequences?

Variable-length input sequences can be handled by padding the shorter sequences so that all sequences in a batch have the same length, and using masking to ensure the RNN ignores the padding token. For better performance, you may also want to create batches containing sequences of similar sizes. [In PyTorch, Chapter 14 does this with packed sequences: pack_padded_sequence() turns a padded batch into a PackedSequence, which recurrent layers like nn.GRU process without running over the padding, and pad_packed_sequence() converts back. The notebook's answer mentions Keras ragged tensors here instead, left over from the book's TensorFlow edition.] Regarding variable-length output sequences, if the length of the output sequence is known in advance (e.g., if you know that it is the same as the input sequence), then you just need to configure the loss function so that it ignores tokens that come after the end of the sequence. Similarly, the code that will use the model should ignore tokens beyond the end of the sequence. But generally the length of the output sequence is not known ahead of time, so the solution is to train the model so that it outputs an end-of-sequence token at the end of each sequence.

## 4. What is beam search, and why would you use it? What tool can you use to implement it?

Beam search is a technique used to improve the performance of a trained Encoder–Decoder model, for example in a neural machine translation system. The algorithm keeps track of a short list of the k most promising output sentences (say, the top three), and at each decoder step it tries to extend them by one word; then it keeps only the k most likely sentences. The parameter k is called the beam width: the larger it is, the more CPU and RAM will be used, but also the more accurate the system will be. Instead of greedily choosing the most likely next word at each step to extend a single sentence, this technique allows the system to explore several promising sentences simultaneously. Moreover, this technique lends itself well to parallelization. [Tool, per Chapter 14: you can write your own (the chapter's notebook has a simple beam_search() function), but in general you'll use the Hugging Face Transformers library, whose generate() method does beam search when you set num_beams to the beam width. The notebook's answer names TensorFlow Addons here instead, left over from the book's TensorFlow edition.]

## 5. What is an attention mechanism? How does it help?

An attention mechanism is a technique initially used in Encoder–Decoder models to give the decoder more direct access to the input sequence, allowing it to deal with longer input sequences. At each decoder time step, the current decoder's state and the full output of the encoder are processed by an alignment model that outputs an alignment score for each input time step. This score indicates which part of the input is most relevant to the current decoder time step. The weighted sum of the encoder output (weighted by their alignment score) is then fed to the decoder, which produces the next decoder state and the output for this time step. The main benefit of using an attention mechanism is the fact that the Encoder–Decoder model can successfully process longer input sequences. Another benefit is that the alignment scores make the model easier to debug and interpret: for example, if the model makes a mistake, you can look at which part of the input it was paying attention to, and this can help diagnose the issue. An attention mechanism is also at the core of the Transformer architecture, in the Multi-Head Attention layers.

## 6. When would you need to use sampled softmax?

Sampled softmax is used when training a classification model when there are many classes (e.g., thousands). It computes an approximation of the cross-entropy loss based on the logit predicted by the model for the correct class, and the predicted logits for a sample of incorrect words. This speeds up training considerably compared to computing the softmax over all logits and then estimating the cross-entropy loss. After training, the model can be used normally, using the regular softmax function to compute all the class probabilities based on all the logits.
