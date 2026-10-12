# Chapter 14 — Natural Language Processing with RNNs and Attention, to fall asleep to

Chapter fourteen. Natural language processing with RNNs and attention.

---

Language and machine intelligence.

In 1950, Alan Turing proposed testing whether a machine can match human intelligence with a linguistic task: a chatbot must fool a human interrogator, chatting by text, into thinking it is human. Turing called it the imitation game. The test has weaknesses. Hardcoded rules can fool naive humans, for example with vague canned answers to keywords, by pretending to joke, or by answering questions with questions, and the test ignores skills such as reading facial expressions or learning a manual task. But it highlights that mastering language is arguably the greatest cognitive ability of Homo sapiens.

Until recently, nearly all state-of-the-art natural language processing models, or NLP models, were recurrent neural networks, or RNNs. Transformers have since replaced them, but RNNs still make transformers easier to understand, and most of the techniques used with them carry over: tokenization, beam search, and attention mechanisms. RNNs have also returned in the form of state space models.

The chapter has three parts. First, a character RNN, or char-RNN, that predicts the next character and generates text. Second, sentiment analysis of movie reviews, which introduces tokenizers and the Hugging Face libraries: Datasets, Tokenizers, and Transformers. Third, an encoder-decoder model that translates English to Spanish, improved with beam search and attention mechanisms. Attention mechanisms learn which part of the inputs to focus on at each time step, and they led directly to transformers.

---

Generating Shakespearean text with a character RNN.

In a 2015 blog post titled "The Unreasonable Effectiveness of Recurrent Neural Networks", Andrej Karpathy trained an RNN to predict the next character in a sentence, then used it to generate text one character at a time. Trained on Shakespeare's works, it learned words, grammar, and proper punctuation, just from predicting the next character. A model that predicts the next piece of text is a language model.

Neural networks need numbers, so text is split into tokens, and each possible token gets an integer ID. Here the tokens are characters. The text, about a quarter of Shakespeare's works, is lowercased to shrink the vocabulary, which leaves thirty-nine distinct characters. Each character's ID is its index in the sorted vocabulary, and two helper functions encode text into a tensor of IDs and decode IDs back into text.

As with the time series in chapter thirteen, the long sequence of characters is cut into windows for a sequence-to-sequence RNN, where each target is the window shifted one character into the future. For the window "to be or not to b", the target is "o be or not to be". Windows are fifty characters long. About one million characters, roughly ninety percent, go to training, and about sixty thousand each to validation and testing. Each training batch holds five hundred twelve windows, reshuffled every epoch. Shorter windows train faster, but the model cannot learn any pattern longer than the window, so it should not be too short.

Raw token IDs work poorly as inputs, because models assume that similar inputs mean similar things, while neighboring IDs can stand for unrelated tokens. One-hot vectors are all equally far apart, but each one is as long as the vocabulary. That is manageable for thirty-nine characters, but not for a word vocabulary of tens of thousands. Embeddings are the better option.

---

Embeddings.

An embedding is a dense representation of higher-dimensional data, usually a categorical feature. With fifty thousand categories, one-hot encoding gives a fifty-thousand-dimensional sparse vector, almost all zeros, while an embedding might have just three hundred dimensions. The embedding size is a hyperparameter, and a rule of thumb is to make it close to the square root of the number of categories.

Embeddings start out random and are trained by gradient descent with the rest of the model. In the California housing dataset, the "near bay" and "near ocean" categories of the ocean proximity feature would start as random vectors, but since they are similar, training tends to push them together and away from "inland". Better representations make accurate predictions easier, so training turns embeddings into useful representations of the categories. This is called representation learning. Learned embeddings can often be reused for other tasks, and in NLP it is often better to reuse pretrained word embeddings than to train new ones.

In 2013, Tomáš Mikolov and other Google researchers published an efficient way to learn word embeddings on a very large corpus, with a neural network trained to predict the words near any given word. Synonyms got very close embeddings, and related words such as France, Spain, and Italy clustered together. The embedding space also has meaningful axes. King minus man plus woman lands very close to queen, which suggests that gender is encoded, and Madrid minus Spain plus France lands near Paris, which suggests the notion of a capital city is encoded too. But embeddings are also noisy and hard to interpret, and they can capture biases: they seem to learn that man is to doctor as woman is to nurse. A 2019 paper by Malvina Nissim and colleagues argued that this example is exaggerated, but fairness in deep learning remains an active research topic.

PyTorch's embedding layer holds a matrix with one row per category, such as one per token in the vocabulary, and one column per embedding dimension, initialized randomly. Embedding a category ID simply looks up its row. Integer inputs of shape batch size by sequence length become floats of shape batch size by sequence length by embedding size. This is mathematically equivalent to one-hot encoding followed by a linear layer without bias, since a one-hot vector selects a single row of the weight matrix. The lookup gives the same result without building one-hot vectors and without all the multiplications by zero.

---

Building and training the char-RNN.

The model has three parts. An embedding layer maps each character ID to ten dimensions. A two-layer GRU with one hundred twenty-eight units per layer and a little dropout processes the sequence, with the batch-first option set, since otherwise the layer expects time before batch. A linear output layer with thirty-nine units gives a logit for every possible character at every time step. The cross-entropy loss and the accuracy metric expect the class dimension second, so the last two dimensions of the output are swapped. The GRU's final hidden states are ignored. Training on Colab with a GPU takes a few hours.

To predict the next character after "To be or not to b", the text is encoded, given a batch dimension of size one, and passed to the model in evaluation mode, without gradient tracking. The argmax of the logits at the last time step gives the predicted character, which is correctly "e".

---

Generating text.

Greedy decoding repeatedly appends the most likely next character and feeds the extended text back to the model. In practice it repeats the same words over and over. Sampling the next character from the model's estimated distribution, so that a character with probability p is picked with probability p, gives more diverse and interesting text. PyTorch's multinomial function does the sampling.

The temperature controls diversity: the logits are divided by it before the softmax. A temperature near zero favors high-probability characters, and a high temperature makes all characters nearly equally likely. Low temperatures suit rigid, precise text such as mathematical equations, and high temperatures suit creative text. Starting from "To be or not to b", a temperature of one hundredth produces text that repeats itself, a temperature of zero point four produces fairly convincing text, and a temperature of one hundred produces gibberish.

Top-k sampling samples only from the k most likely characters. Top-p sampling, also called nucleus sampling, samples only from the smallest set of most likely characters whose total probability exceeds a threshold. Both give more convincing text. Beam search, more layers and neurons, longer training, and more regularization also help. The model cannot learn patterns longer than its fifty-character window, and even LSTM and GRU cells cannot handle very long sequences.

Predicting the next character also requires higher-level understanding. After "Great movie, I really", the letter "l", for "loved", is likelier than "h", for "hated". In 2017, Alec Radford and other OpenAI researchers trained a big char-RNN-like model on a large dataset and found that one of its neurons was an excellent sentiment classifier. This sentiment neuron reached state-of-the-art results on sentiment benchmarks without any labels, which foreshadowed unsupervised pretraining in NLP.

---

Sentiment analysis and the IMDb dataset.

Sentiment analysis predicts how positive or negative a text is, and it is one of the most common text classification tasks. If MNIST is the "Hello, world" of computer vision, sentiment analysis on the IMDb reviews dataset is the "Hello, world" of NLP. The dataset has fifty thousand English movie reviews from the Internet Movie Database, split evenly between training and testing, each labeled zero for negative or one for positive. It is small enough for a laptop and hard enough to be rewarding.

The Hugging Face Datasets library downloads it from the Hugging Face Hub, which hosts hundreds of thousands of datasets, and its train test split method keeps eighty percent of the original training set for training and twenty percent for validation. Some reviews are easy: one calls "The Rookie" a wonderful movie in its first sentence. Others mix praise and criticism through a long description and only become clearly negative in their conclusion. A simple char-RNN would struggle, so a better tokenization is needed.

---

Subword tokenization and byte pair encoding.

A token is a small piece of text from a fixed-size vocabulary: a word, a subword, a character, or a byte. A tokenizer splits text into tokens. Subword tokenization, explored in a 2016 paper by Rico Sennrich and colleagues at the University of Edinburgh, allows a model to handle rare words it has never seen. A model that knows "smart", and knows that the suffix "est" means "the most", can infer the meaning of "smartest". One method they evaluated is byte pair encoding, or BPE, introduced by Philip Gage in 1994 for data compression. BPE splits the whole training set into characters, then repeatedly finds the most frequent pair of adjacent tokens and adds it to the vocabulary, until the vocabulary reaches the desired size.

The Hugging Face Tokenizers library has fast implementations of BPE and other algorithms. The IMDb tokenizer is a BPE model with an unknown token, which replaces anything never seen during training. It uses a whitespace pre-tokenizer, which splits text at spaces, drops the spaces, and separates groups of letters from groups of other characters, so "Hello, world" followed by three exclamation marks becomes four chunks: "Hello", the comma, "world", and the three exclamation marks. Running BPE within these chunks speeds up training and gives better tokens for English. The trainer has a vocabulary limit of one thousand tokens and puts two special tokens first: padding, with ID zero, and unknown, with ID one. Training on the lowercased reviews takes a few seconds.

Encoding "what an awesome movie" followed by an emoji gives eight tokens: "what", "an", "aw", "es", "ome", "movie", the exclamation mark, and the unknown token, which replaces the emoji. Frequent words get a single token, and rarer words like "awesome" are split. The tokenizer can also decode IDs back into text, and it records each token's start and end offsets in the original string, which helps with debugging.

To stack a batch of encodings into one tensor, padding is enabled, so shorter sequences are padded to the length of the longest one in the batch, and truncation is enabled at five hundred tokens, since RNNs handle very long sequences poorly. Each encoding has an attention mask, with a one for each real token and a zero for each padding token. Multiplying a tensor by the mask zeroes out the padded steps, and summing the mask gives each sequence's length.

---

Byte-level BPE, WordPiece, and Unigram LM.

Because the whitespace pre-tokenizer drops spaces, decoding puts a space between every pair of tokens, so "awesome" comes back as "aw es ome". The byte-level pre-tokenizer instead replaces each space with a special character, a capital G with a dot above it, so the spaces can be restored. It also makes BPE work on bytes rather than characters, which is called byte-level BPE, or BBPE. An emoji becomes four bytes in the UTF-8 encoding. If the vocabulary contains all two hundred fifty-six possible bytes, BBPE never needs the unknown token, since any text can fall back to single bytes. This suits text with rare characters such as emojis.

WordPiece, proposed by Google in 2016, is like BPE, but at each iteration it adds the pair with the highest score instead of the most frequent pair. The score is the pair's frequency, divided by the product of the frequencies of its two tokens, times the vocabulary size. The denominator penalizes pairs whose tokens are already frequent on their own. This favors more useful, meaningful tokens and often gives shorter sequences than BPE. WordPiece marks tokens that continue a word with a prefix of two hash signs, which makes the original text easy to rebuild.

Unigram LM, short for unigram language model, was introduced by Taku Kudo at Google in 2018. It works the other way around. It starts with a very large vocabulary of every frequent word, subword, and character, and gradually removes the least useful tokens. It assumes that the corpus was produced by sampling tokens one at a time, each independently of the others. That makes it possible to estimate the probability of the whole corpus, and at each iteration it removes tokens without reducing that probability much.

For example, suppose "them" has probability one percent, "the" five percent, and "m" two percent. The word "them" can be sampled as one token, with a one percent chance, or as "the" then "m", with five percent times two percent, which is a tenth of a percent, for a total of one point one percent. Removing the token "them" drops that to a tenth of a percent, while removing "the" or "m" only drops it to one percent. If the corpus contained only the word "them", the algorithm would drop "the" or "m". In a real corpus, it would likely find less useful tokens to drop. Unigram LM works well for languages that do not separate words with spaces the way English does: Chinese has no spaces between words, Vietnamese has spaces even within words, and German often joins words together.

The same paper introduced subword regularization: while training the NLP model, not the tokenizer, the same word is randomly tokenized in different ways, such as "them" or "the" plus "m". This improves generalization and robustness, especially for morphologically rich languages, whose words carry grammar through affixes and inflections, such as Arabic, Finnish, German, Hungarian, Polish, and Turkish. The Tokenizers library does not support it, but Google's open source SentencePiece library does.

To compare the three main tokenizers: BBPE merges the most frequent pairs. It is fast, simple, and great for multilingual text, but can produce awkward splits, and GPT, Llama, RoBERTa, and BLOOM use it. WordPiece merges the pairs that maximize the data's likelihood. It balances efficiency and token quality but is less robust for multilingual text, and BERT, DistilBERT, and ELECTRA use it. Unigram LM removes the least likely tokens. It gives the most meaningful tokens and the shortest sequences but is slower to train and to tokenize, and T5, ALBERT, and mBART use it.

Training a custom tokenizer makes sense for jargon-heavy domains such as medicine, law, or engineering, for low-resource languages or dialects, or for a new programming language. In most cases, a pretrained tokenizer is reused.

---

Reusing pretrained tokenizers.

The Hugging Face Transformers library provides many pretrained models for NLP, computer vision, audio, and more, and downloads their weights automatically from the Hub. They were originally all transformers, hence the name, but other kinds such as CNNs are now included. Each model comes with what it needs, including its tokenizer.

The library's auto tokenizer class downloads a tokenizer by checkpoint name. GPT-2, a text generation model, uses a BBPE tokenizer. Called like a function on a list of texts, a tokenizer returns a dictionary-like batch encoding containing the input IDs, and its decode method turns IDs back into text.

BERT, short for Bidirectional Encoder Representations from Transformers, uses WordPiece. Unlike GPT-2's tokenizer, BERT's has a padding token, so it can pad a batch and return PyTorch tensors of token IDs and attention masks directly. The checkpoint "bert-base-uncased" is a case-insensitive BERT pretrained on English. Other checkpoints include a larger case-sensitive model and an uncased multilingual model covering over one hundred languages. BERT's tokenizer adds a class token, written CLS, at the start of each sequence and a separator token, written SEP, at the end. BERT needs them, and they can be turned off for models that do not.

ALBERT and T5 use Unigram LM tokenizers, which load the same way. The Transformers library can also wrap a custom tokenizer from the Tokenizers library so that it has the same interface as the pretrained ones.

---

Building a sentiment analysis model.

Tokenization is done in the data loader, through its collate function, which the loader calls on every batch. The function tokenizes the reviews with the BERT tokenizer, padding them and truncating them to two hundred tokens, and returns the tensors of token IDs and attention masks along with a tensor of labels. Batches hold two hundred fifty-six reviews.

The model is like the Shakespeare model, with a few differences. The embedding layer's padding index is set, so padding tokens get a fixed, nontrainable zero vector, which reduces their effect on the loss. It is a sequence-to-vector model, so only the top GRU layer's last hidden state is used. The output layer has a single unit giving one logit per review, positive for positive reviews and negative for negative ones. Trained with the binary cross-entropy with logits loss, it reaches close to eighty-five percent validation accuracy. The best models reach human level, slightly above ninety percent. Higher is probably impossible, because many reviews are so ambiguous that classifying them is like flipping a coin.

---

Packed sequences.

The GRU still processes every padding token, and after a long run of padding it may have forgotten what the review said. A packed sequence fixes this. It is a data structure that represents a batch of variable-length sequences efficiently. PyTorch's pack padded sequence function builds one from a padded tensor and the sequence lengths, and the pad packed sequence function converts it back. By default, packing assumes the sequences are sorted from longest to shortest and that the time dimension comes before the batch dimension, and options handle unsorted and batch-first data.

PyTorch's recurrent layers accept packed sequences and stop each sequence at its true end. The sentiment model computes the lengths by summing the attention mask, packs the embeddings, and passes them to the GRU, so padding no longer has any effect. A recurrent layer given a packed sequence also returns packed outputs, which must be converted back to a padded tensor before other layers can use them.

---

Bidirectional RNNs.

A regular recurrent layer is causal: at each step it sees only past and present inputs. That suits forecasting and the decoder of a sequence-to-sequence model. For text classification and for encoders, it is better to look ahead, since the meaning of "right" in "the right arm", "the right person", and "the right to speak" depends on the words after it. A bidirectional recurrent layer runs one recurrent layer left to right and another right to left over the same inputs, and concatenates their outputs at each step. In PyTorch it is an option on the GRU, LSTM, and simple RNN modules.

The output layer's input size doubles. The hidden states tensor now holds a forward and a backward state for each layer, so the model takes the top layer's two states and concatenates them into one vector per review, twice the hidden size. Per-step outputs of a bidirectional layer are also twice as wide.

Here it does not help. The first model already overfit the training set, and the bidirectional version overfits more: over ninety-nine percent training accuracy, but eighty-four percent on validation. More regularization, a smaller model, or more data would help. Another option is pretrained embeddings.

---

Reusing pretrained embeddings.

Embeddings trained on a huge corpus are better than ones learned from twenty-five thousand reviews, and they transfer across topics and tasks, since "amazing" means much the same everywhere. The model already uses BERT's tokenizer, so it can use BERT's embeddings. The Transformers library's auto model class downloads pretrained BERT, a PyTorch model whose word embedding layer has thirty thousand five hundred twenty-two tokens of seven hundred sixty-eight dimensions. The sentiment model's own embedding layer is initialized with a copy of that matrix and frozen, so that large gradients early in training cannot damage it. After a few epochs, it can be unfrozen to fine-tune the embeddings for the task.

Well-known pretrained word embeddings include Google's Word2vec from 2013, Stanford's GloVe from 2014, and Facebook's FastText from 2016. Their limit is that each word has one representation regardless of context, so "right" is encoded the same way in "left and right" and in "right and wrong".

---

Pretrained language models.

In 2018, Matthew Peters and colleagues introduced Embeddings from Language Models, or ELMo: contextualized word embeddings taken from the internal states of a deep bidirectional RNN language model. A model can reuse several layers of a pretrained language model, not just its word embeddings.

At about the same time, the Universal Language Model Fine-Tuning paper, or ULMFiT, by Jeremy Howard and Sebastian Ruder, trained an LSTM language model on a huge corpus with self-supervised learning, generating labels automatically from the data, then fine-tuned it on various tasks. It beat the state of the art on six text classification tasks, cutting error rates by eighteen to twenty-four percent in most cases. Fine-tuned on just one hundred labeled examples, it matched a model trained from scratch on ten thousand. Before this, pretraining was the norm only in computer vision, and NLP pretraining was limited to word embeddings. Today, reusing pretrained language models is the norm.

The whole BERT model can be reused. Its last hidden state holds a contextualized embedding for every token, with shape batch size by sequence length by seven hundred sixty-eight. Here "last" means the last layer, not the last time step, since BERT is a transformer, not an RNN. Feeding these embeddings to a GRU, which no longer needs to be bidirectional because the embeddings have already looked ahead, and training only the GRU and output layer with BERT frozen, gives over eighty-eight percent accuracy.

Simpler still, the output layer can use only the contextualized embedding of the first token, the class token. During pretraining, BERT had to classify texts from that token's embedding alone, so it learned to summarize the most important features of the text there, and the GRU can be dropped. BERT also has an extra hidden layer on top of the class token's embedding, a linear layer with a hyperbolic tangent activation, called the pooler. Its output can be used instead, and the pooler can be unfrozen after a few epochs to fine-tune it.

---

Task-specific classes.

The Transformers library also has ready-made task models. BERT for sequence classification is pretrained BERT plus a classification head, created from a checkpoint name, the number of labels, and optionally a data type, such as sixteen-bit floats to fit small GPUs. Others include BERT for question answering and RoBERTa for sequence classification, and the auto model for sequence classification class picks the right one for a given checkpoint.

Hugging Face treats binary classification exactly like multiclass classification. The number of labels is two, the model outputs two logits, training uses cross-entropy rather than binary cross-entropy, and probabilities come from the softmax rather than the sigmoid. The classification head starts untrained, so before fine-tuning the model calls "This was a great movie" negative with about sixty-six percent confidence. If labels are passed in, the model also returns the loss: cross-entropy for two or more labels, or mean squared error for a single label, which suits regression.

---

The Trainer API.

The Trainer API fine-tunes models with very little boilerplate. It handles checkpoints, early stopping, multiple GPUs, metric logging, padding, batching, and shuffling. It takes dataset objects rather than data loaders, and they must already be tokenized, which the Datasets library's map method does, in batches for speed. A small function computes the accuracy from the label IDs and the predictions. Hugging Face's Evaluate library also provides metrics.

A training arguments object sets the configuration: an output directory, two epochs, a batch size of one hundred twenty-eight, evaluation, logging, and saving at the end of each epoch, and loading the best model by validation accuracy at the end. Logs can be reported to tools such as TensorBoard or Weights and Biases. A trainer object takes the model, the arguments, the training and validation sets, the metric function, and a data collator that pads each batch. Calling its train method reaches about ninety percent validation accuracy after two epochs.

---

Hugging Face pipelines.

A pipeline bundles a pretrained model with its preprocessing and post-processing, ready to use with no training. A sentiment analysis pipeline returns, for each text, a label, positive or negative, and a score, which is the estimated probability of that label. Its default model, at the time of writing, is DistilBERT, a scaled-down BERT with an uncased tokenizer, trained on English Wikipedia and a corpus of English books, and fine-tuned on the Stanford Sentiment Treebank version two, or SST-2. It reaches eighty-eight point two percent accuracy on the IMDb validation set. Pipelines use a GPU automatically, and Transformers models are in evaluation mode by default. Since the score is for the chosen label, in binary classification it is never below one half.

The Hugging Face Hub has over eighty thousand text classification models, so filtering by task, popularity, language, and dataset is essential, and models from reputable sources are preferable. A model's remote code should only be allowed to run if its author is trusted. A model can contain malicious code, produce biased outputs, leak copyrighted or private training data, or have poisoned weights that produce harmful content, such as propaganda, only for certain inputs.

Natural language inference is another text classification task. A model fine-tuned on the MultiNLI dataset classifies a pair of texts, separated by separator tokens, as contradiction, entailment, or neutral. "She loves me" and "She loves me not" contradict each other. "Alice just woke up" entails "Alice is awake". "I like dogs" and "Everyone likes dogs" are neutral. Pipelines also cover question answering, summarization, translation, text generation, and many vision and audio tasks, such as object detection and speech-to-text.

---

Bias and fairness.

The sentiment pipeline rates "I am from the USA" very positive and "I am from Iraq" very negative, and it is positive about Thailand but negative about Vietnam. The bias comes largely from training data full of references to the wars in Iraq and Vietnam, and fine-tuning with only two classes amplified it. A model fine-tuned with an extra neutral class mostly loses the country bias.

The architecture, the loss, the regularization, and the optimizer can also introduce bias. A model that will be deployed, such as one that scores resumes, should be evaluated across subsets of people, not just on average, since it may work well overall and terribly for some groups. Counterfactual tests check that predictions do not change when only someone's gender or place of birth changes. Fixes include rebalancing the dataset, fine-tuning on other data, switching to another pretrained model, or adjusting the architecture or hyperparameters. Even a perfectly fair model can be used in a biased way, for example by recruiters who apply it to only some categories of people.

---

An encoder-decoder network for neural machine translation.

The translation model is a sequence-to-sequence model, following a 2014 paper by Ilya Sutskever and colleagues. Token IDs pass through an embedding layer, the encoder reads the English text, and the decoder outputs the Spanish translation. During training, the decoder's inputs are the target translation shifted one step back, so at each step it receives the token it should have output at the previous step, whatever it actually output. This is called teacher forcing, and it speeds up training and improves performance. The decoder starts from a start-of-sequence token and must end with an end-of-sequence token.

At each step, the decoder's output layer gives a logit for every token in the vocabulary, and a softmax turns them into probabilities, so training is a classification task with cross-entropy loss. At inference time there is no target, so the decoder is fed its own previous output. In 2015, Samy Bengio and colleagues proposed gradually switching from target tokens to output tokens during training.

---

The translation dataset and model.

The data is the English-Spanish part of the Tatoeba Challenge dataset, which University of Helsinki researchers built from the Tatoeba language-learning project to benchmark machine translation. Its training set is very large, so its validation set is used for training, with twenty percent held out for validation. English and Spanish share many words, such as hotel, idea, and radio, and subwords, such as pre and auto, so a single BPE tokenizer is trained on both languages, with ten thousand tokens and special tokens for padding, unknown, start of sequence, and end of sequence.

The collate function adds the start and end tokens to each Spanish text and tokenizes both languages. The decoder input drops the end token, and the target drops the start token. For example, the input is the start token followed by "Me gusta el fútbol", and the target is "Me gusta el fútbol" followed by the end token.

The encoder and decoder share one embedding layer, and each is a two-layer GRU with five hundred twelve units. The encoder reads the packed English embeddings, and its final hidden states, for both layers, become the decoder's initial hidden states. The decoder reads the Spanish embeddings, and a linear layer outputs logits over the vocabulary. The Spanish side is not packed. Instead, the cross-entropy loss is set to ignore positions where the target is padding. Training takes a while, which is reasonable for a model learning two languages at once.

The most common translation metric is the bilingual evaluation understudy score, or BLEU. It compares each translation with several good human translations, counting the n-grams, meaning sequences of n words, that appear in any of them, adjusted for how often they occur.

---

Translating with the model.

To translate, the model receives the English text and a single start token, the most likely next token is appended to the decoder input, and the process repeats until the model outputs the end token. "I like soccer" becomes "Me gusta el fútbol". Longer sentences fail: "I like to play soccer with my friends" becomes "Me gusta jugar con mis amigos", which means "I like to play with my friends", with no soccer. A larger training set and more GRU layers would help. The encoder can be made bidirectional, but the decoder cannot, because it must stay causal: otherwise it would see the full translation instead of only the previous tokens. Beam search also helps greatly at inference time.

---

Speeding up large output layers.

With ten thousand tokens or more in the output vocabulary, computing the loss can be slow. Sampled softmax, from Sébastien Jean and colleagues in 2015, approximates the softmax using the correct class's logit and a random sample of other logits. It needs the target, so it only works during training, and PyTorch does not include it. Adaptive softmax, from Edouard Grave and colleagues in 2016, splits the vocabulary into clusters by frequency. Frequent tokens are handled normally and rarer ones go into progressively larger clusters, so only the needed clusters are computed, speeding up both training and inference. PyTorch implements it.

Weight tying, proposed by Ofir Press and Lior Wolf in 2016, reuses the embedding matrix as the output layer's weights, optionally with the output bias removed. It saves memory, cuts the number of parameters, speeds up training, and can improve accuracy when training data is limited. The embedding layer maps one-hot vectors into the embedding space, and the output layer does the reverse. So if the embedding matrix is orthogonal, meaning its transpose equals its inverse, separate output weights are unnecessary.

---

Beam search.

The model produces one word at a time, so one mistake can trigger more. For "I like soccer", the model might correctly start with "Me" but then choose "gustan", the plural, instead of "gusta", which is understandable since "Me gustan" often starts a translation of "I like". That leads it to "los", the plural for "the", and since it never saw "los fútbol", to "jugadores": "I like the players". One error caused a chain of errors.

Beam search keeps a short list of the k most promising partial translations, where k is the beam width. At each step it extends each one by every possible token and keeps only the k most likely results. A candidate's probability is the probability of its prefix times the model's conditional probability for the new token.

With a beam width of three, the first step might keep "Me" at seventy-five percent, "a" at three percent, and "como" at one percent. With ten thousand tokens, the next step scores thirty thousand two-word candidates. "Me gustan" gets seventy-five percent times thirty-six percent, which is twenty-seven percent, and the top three are "Me gustan", "Me gusta" at twenty-four percent, and "Me encanta" at twelve percent. "Me gustan" leads, but "Me gusta" survives. Two steps later, "Me gusta el fútbol" leads at six percent, and "Me gustan" is gone. The model improved without any extra training, just by being used more wisely.

The Transformers library's generate method supports beam search through its number of beams argument, random sampling through its sampling option, and other generation strategies that can be combined. With beam search, "I like to play soccer with my friends" is translated correctly. But adding "at the beach" produces "I like to play with play with the players of the beach". The limited short-term memory of RNNs is still the problem.

---

Attention mechanisms.

The representation of "soccer" has to travel through many steps before the decoder outputs "fútbol". A landmark 2014 paper by Dzmitry Bahdanau and colleagues allowed the decoder to focus, at each step, on the appropriate words as encoded by the encoder, so the path from each word to its translation becomes short and the RNN's short-term memory matters much less. Attention revolutionized neural machine translation and deep learning in general, especially for sentences of over thirty words.

The encoder, generally made bidirectional, sends all of its outputs to the decoder, not just its final hidden state. At each step, the decoder computes a weighted sum of those outputs. The weight alpha t i is the weight of the i-th encoder output at the t-th decoder step, and a large weight on the output for "soccer" means the decoder focuses on that word at that step. The decoder still also receives its previous hidden state and the previous target token.

The weights come from a small network called an alignment model, or attention layer, trained jointly with the rest of the model. A dense layer takes each encoder output together with the decoder's previous hidden state and outputs a score, also called an energy, measuring how well they align. If the decoder has output "me gusta el" and now expects a noun, "soccer" scores highest. A softmax turns the scores into weights that add up to one. This is Bahdanau attention. Because it concatenates the encoder output with the decoder state, it is also called concatenative attention, or additive attention.

---

Attention as memory retrieval.

Attention also works as a differentiable memory lookup. Suppose the encoder has worked out that "I" is the subject, "like" is the verb, and "soccer" is the noun. The decoder needs the noun next, as if looking up the key "noun" in a dictionary. But keys and queries are learned vectors, so they never match exactly. Instead, attention computes the similarity between the query and every key, turns the similarities into weights with a softmax, and returns the weighted sum of the values. If the noun key is by far the most similar, the result is very close to the representation of "soccer". The decoder asked for a noun, and attention retrieved it.

The arguments are called query, key, and value. In this model, the query is the decoder's hidden states, and the keys and values are both the encoder outputs: the keys for computing the weights, the values for the weighted sum. With input and output sentences of about n words, attention computes about n squared weights, and this quadratic cost becomes intractable for very long sentences.

---

Luong attention.

In 2015, Minh-Thang Luong and colleagues proposed multiplicative attention, or Luong attention, which scores each encoder output by its dot product with the decoder's hidden state. The dot product is a fairly good similarity measure and very fast on modern hardware, but both vectors must have the same dimensionality. They also used the decoder's current hidden state rather than the previous one, and concatenated the resulting attention vector with that hidden state to predict the next token. This keeps attention out of the decoder's recurrence, which is simpler and faster.

A variant called the general dot product first passes the encoder outputs through a linear layer without bias. Both dot product variants outperformed concatenative attention, which is now much less used.

In words, the attention vector at decoder step t is the sum of all the encoder outputs, each multiplied by its weight, and the weights are the softmax of the scores for that step. The dot score is the dot product of the decoder state and the encoder output. The general score inserts a learned weight matrix between them. The concatenative score is the dot product of a learned vector with the hyperbolic tangent of a weight matrix times the decoder state and encoder output concatenated together.

---

Adding attention to the translation model.

PyTorch has no Luong attention function, but a short one does the job. The query, of shape batch size by query length by dimension, is multiplied by the transposed keys, giving scores of shape batch size by query length by key length. A softmax over the last dimension turns the scores into weights, which multiply the values to give an output of shape batch size by query length by value dimension. Keys and values are usually identical, though some models use compressed keys to save memory and time.

In the model, the encoder's packed outputs are converted back to a padded tensor. The decoder outputs serve as the query, since they equal its top layer's hidden states at each step, and the encoder outputs serve as keys and values. The attention output is concatenated with the decoder output, so the output layer's input size doubles. This version does not mask padding tokens. The model learns to ignore them, but masking them is preferable, as chapter fifteen shows.

With attention, the model translates "I like to play soccer with my friends at the beach" correctly, even without beam search. Attention proved so powerful that Google researchers dropped recurrent layers entirely, keeping only attention and feedforward layers. It worked remarkably well, and their paper, "Attention Is All You Need", introduced the Transformer architecture.

---

Summary.

Text is split into tokens, each token gets an integer ID, and an embedding layer maps each ID to a trainable dense vector. Embeddings learned on large corpora capture meaning and can be reused, but they can also capture biases.

A char-RNN trained to predict the next character is a small language model. It generates text one token at a time, by greedy decoding or by sampling, with the temperature, top-k sampling, or top-p sampling controlling diversity.

Subword tokenizers handle rare and unseen words. BPE merges the most frequent pairs, byte-level BPE never needs an unknown token, WordPiece merges the highest-scoring pairs, and Unigram LM prunes a large vocabulary. Padding, truncation, and attention masks turn variable-length texts into batches, and packed sequences let recurrent layers skip the padding.

Bidirectional recurrent layers use context on both sides of each word, which suits classifiers and encoders but not decoders. Pretrained language models such as BERT provide contextualized embeddings, and fine-tuning a pretrained model is now the norm. The Hugging Face libraries provide datasets, tokenizers, pretrained models, task-specific classes, the Trainer API, and pipelines. Models can carry biases and security risks, so they should be evaluated across groups and downloaded only from trusted sources.

An encoder-decoder model translates by encoding the source sentence and decoding one token at a time, trained with teacher forcing. Beam search keeps several candidate translations so the model can recover from early mistakes. Attention gives the decoder a weighted average of all the encoder outputs at each step, with weights from comparing a query to keys. It overcomes the short memory of RNNs on long sentences, and it led directly to the Transformer.
