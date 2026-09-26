# Chapter 13 — Processing Sequences Using RNNs and CNNs

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. Answers below
are the author's, from the exercise solutions at the end of
`handson-mlp-main/13_processing_sequences_using_rnns_and_cnns.ipynb`,
converted to plain text for the flashcard app.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text. Solutions are in the same
notebook unless noted.

- 9. Tweak the Seq2Seq model to forecast both rail and bus ridership for the
  next 14 days (28 outputs instead of 14).
- 10. Train a model on the Bach chorales dataset to predict the next time
  step (four notes), then use it to generate Bach-like music one note at a
  time.
- 11. Train a sketch classifier on the QuickDraw dataset: concatenate each
  sketch's pen strokes into one sequence, plus a stroke-progress feature. No
  author solution yet.
- 12. Record yourself saying "yes" and "no", split the recordings into words
  with torchaudio's Vad, convert them to mel spectrograms, and train a
  binary classification RNN. No author solution yet.

## 1. Can you think of a few applications for a sequence-to-sequence RNN? What about a sequence-to-vector RNN, and a vector-to-sequence RNN?

Here are a few RNN applications:
- For a sequence-to-sequence RNN: predicting the weather (or any other time series), machine translation (using an Encoder–Decoder architecture), video captioning, speech to text, music generation (or other sequence generation), identifying the chords of a song
- For a sequence-to-vector RNN: classifying music samples by music genre, analyzing the sentiment of a book review, predicting what word an aphasic patient is thinking of based on readings from brain implants, predicting the probability that a user will want to watch a movie based on their watch history (this is one of many possible implementations of collaborative filtering for a recommender system)
- For a vector-to-sequence RNN: image captioning, creating a music playlist based on an embedding of the current artist, generating a melody based on a set of parameters, locating pedestrians in a picture (e.g., a video frame from a self-driving car's camera)

## 2. How many dimensions must the inputs of an RNN layer have? What does each dimension represent? What about its outputs?

An RNN layer must have three-dimensional inputs: if you set batch_first=True when creating the RNN layer, the first dimension is the batch dimension (its size is the batch size), the second dimension represents the time (its size is the number of time steps), and the third dimension holds the inputs at each time step (its size is the number of input features per time step). For example, if you want to process a batch containing 5 time series of 10 time steps each, with 2 values per time step (e.g., the temperature and the wind speed), the shape will be [5, 10, 2]. The outputs are also three-dimensional, with the same first two dimensions, but the last dimension is equal to the number of neurons. For example, if an RNN layer with 32 neurons processes the batch we just discussed, the output will have a shape of [5, 10, 32]. If you set batch_first=False (which is the default), then the first two dimensions are swapped.

## 3. How can you build a deep sequence-to-sequence RNN in PyTorch?

To build a deep sequence-to-sequence RNN in PyTorch, the simplest and most efficient option is to use the nn.RNN, nn.LSTM, or nn.GRU modules and set the num_layers hyperparameter to the desired number of layers. However, if you need more flexibility (e.g., to add dropout or layer norm between time steps), you must create a custom module and implement the recurrent loop manually. For this, you can use the nn.RNNCell, nn.LSTMCell, or nn.GRUCell modules and run them at each step, passing them both the inputs and the hidden states.

## 4. Suppose you have a daily univariate time series, and you want to forecast the next seven days using an RNN. Which architecture should you use?

If you have a daily univariate time series, and you want to forecast the next seven days, the simplest RNN architecture you can use is a stack of RNN layers, using seven neurons in the output RNN layer. You can then train this model using random windows from the time series (e.g., sequences of 30 consecutive days as the inputs, and a vector containing the values of the next 7 days as the target). This is a sequence-to-vector RNN. Alternatively, you could use a sequence-to-sequence RNN. You can train this model using random windows from the time series, with sequences of the same length as the inputs as the targets. Each target sequence should have seven values per time step (e.g., for time step t, the target should be a vector containing the values at time steps t + 1 to t + 7).

## 5. What are the main difficulties when training RNNs? How can you handle them?

The two main difficulties when training RNNs are unstable gradients (exploding or vanishing) and a very limited short-term memory. These problems both get worse when dealing with long sequences. To alleviate the unstable gradients problem, you can use a smaller learning rate, use a saturating activation function such as the hyperbolic tangent (which is the default), and possibly use gradient clipping, Layer Normalization, or dropout at each time step. To tackle the limited short-term memory problem, you can use LSTM or GRU layers (this also helps with the unstable gradients problem). That said, more recent architectures such as Transformers and State-Space Models (SSMs, see Appendix E) have a much longer memory.

## 6. Can you sketch the LSTM cell’s architecture?

An LSTM cell's architecture looks complicated, but it's actually not too hard if you understand the underlying logic. The cell has a short-term state vector and a long-term state vector. At each time step, the inputs and the previous short-term state are fed to a simple RNN cell and three gates: the forget gate decides what to remove from the long-term state, the input gate decides which part of the output of the simple RNN cell should be added to the long-term state, and the output gate decides which part of the long-term state should be output at this time step (after going through the tanh activation function). The new short-term state is equal to the output of the cell. See Figure 13–12.

## 7. Why would you want to use 1D convolutional layers in an RNN?

An RNN layer is fundamentally sequential: in order to compute the outputs at time step t, it has to first compute the outputs at all earlier time steps. This makes it impossible to parallelize (unless the RNN is fully linear, as in modern SSMs, see Appendix E). On the other hand, a 1D convolutional layer lends itself well to parallelization since it does not hold a state between time steps. In other words, it has no memory: the output at any time step can be computed based only on a small window of values from the inputs without having to know all the past values. Moreover, since a 1D convolutional layer is not recurrent, it suffers less from unstable gradients. One or more 1D convolutional layers can be useful in an RNN to efficiently preprocess the inputs, for example to reduce their temporal resolution (downsampling) and thereby help the RNN layers detect long-term patterns. In fact, it is possible to use only convolutional layers, for example by building a WaveNet architecture.

## 8. Which neural network architecture could you use to classify videos?

To classify videos based on their visual content, one possible architecture could be to take (say) one frame per second, then run every frame through the same convolutional neural network (e.g., a pretrained Xception model, possibly frozen if your dataset is not large), feed the sequence of outputs from the CNN to a sequence-to-vector RNN, and finally run its output through a softmax layer, giving you all the class probabilities. For training you would use cross entropy as the cost function. If you wanted to use the audio for classification as well, you could use a stack of strided 1D convolutional layers to reduce the temporal resolution from thousands of audio frames per second to just one per second (to match the number of images per second), and concatenate the output sequence to the inputs of the sequence-to-vector RNN (along the last dimension). However, transformers and SSMs are much better choices for video classification (see Chapters 15, 16, and Appendix E).
