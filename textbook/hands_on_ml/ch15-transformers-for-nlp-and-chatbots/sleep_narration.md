# Chapter 15 — Transformers for Natural Language Processing and Chatbots, to fall asleep to

Chapter fifteen. Transformers for natural language processing and chatbots.

---

The Transformer.

In a landmark 2017 paper titled "Attention Is All You Need", Google researchers proposed the Transformer, which significantly improved the state of the art in neural machine translation. It is an encoder-decoder model, used much like the recurrent one from chapter fourteen. The encoder turns the source text into one contextualized embedding per token. The decoder receives the encoder's output and the translation so far, starting with a start-of-sequence token, and predicts the next token for each input token. The last predicted token is appended, and the process repeats until an end-of-sequence token appears. During training, the full target is known, so it is fed to the decoder in a single pass.

The Transformer has no recurrent or convolutional layers, only dense layers and a new kind of attention called multi-head attention. Because it is not recurrent, it suffers less from vanishing and exploding gradients, trains in fewer steps, parallelizes easily across GPUs, and scales surprisingly well. Multi-head attention also captures long-range patterns much better than RNNs.

---

The transformer revolution.

2018 was called the "ImageNet moment for NLP". In June, OpenAI released GPT, built only from the Transformer's decoder and pretrained on a large corpus, with unprecedented text generation. A few months later, Google released BERT, built only from the encoder, which excelled at natural language understanding tasks, or NLU tasks, such as text classification and question answering. Transformers also turned out to be great at vision, audio, and robotics. Feeding image patches to a transformer gives a vision transformer, and models that handle several modalities at once, such as text and images, are called multimodal models.

This mix of performance, flexibility, and scalability led Google, OpenAI, Meta, Microsoft, Anthropic, and others to train ever larger transformers. The original had about sixty-five million parameters. By January 2021, models reached one point six trillion, and training one from scratch takes months and typically tens of millions of dollars.

In November 2022, OpenAI released ChatGPT, which reached one million users in five days and over one hundred million monthly users in two months. It used a variant of GPT-3.5 fine-tuned to be conversational, helpful, and safe. Perplexity, Google's Gemini, Anthropic's Claude, Mistral AI, and DeepSeek followed. Google had built a powerful chatbot named LaMDA earlier but kept it private, likely because it was not considered safe enough.

Many of these models are proprietary, available only through a web interface, an app, or a paid API. Others are open weights and can be downloaded for free, for example from the Hugging Face Hub. Some have license restrictions, such as Meta's Llama models. Others are truly open source, such as DeepSeek's R1 and Mistral-7B, and Ai2's OLMo models even include their training code and data.

---

The original Transformer architecture.

The encoder gradually transforms each token's representation until it captures the token's meaning in context. Apart from the embedding layer, every encoder layer takes a tensor of shape batch size by sequence length by embedding size and returns one of the same shape, hence the name. In "I like soccer", the token "like" starts out vague, since "like" means different things in "I'm like a cat" and "I like my cat". After the encoder, it should mean to be fond of, and carry anything else useful for translation, such as being a verb.

The decoder turns each input token's representation into one that predicts the following token. If the decoder input is the start token followed by "me gusta el fútbol", the start token becomes "me", "me" becomes "gusta", and so on, until "fútbol" becomes the end token. A final linear layer outputs a logit for every vocabulary token. In PyTorch, the softmax is not added explicitly, because the cross-entropy loss applies it to the logits.

The encoder and decoder are stacks of N identical blocks, with N equal to six in the paper, and the encoder stack's final output feeds every decoder block. Most components are familiar: embedding layers, skip connections each followed by layer normalization, feedforward modules of two dense layers, the first with ReLU, and a linear output layer. All of these treat each token independently, so new components are needed to relate tokens to each other.

The encoder's multi-head attention layer updates each token by attending to every token in the same sentence, including itself. This is self-attention, and it is where "like" gets its precise meaning, for example by noticing the subject "I". The decoder's masked multi-head attention does the same but is causal: a token cannot attend to later tokens. Processing "gusta", it sees only the start token, "me", and "gusta", or the model could cheat during training. The decoder's upper multi-head attention layer attends to the encoder's outputs, which is called cross-attention. Processing "el" to produce "fútbol", it will likely attend closely to "soccer". In self-attention, the queries, keys, and values all come from the previous layer. In cross-attention, the keys and values are the encoder's final outputs, and the queries come from the previous decoder layer.

---

Positional encodings.

Attention and dense layers are position-agnostic: they have no idea where a token sits in the sentence. Word order matters, so a positional encoding, a dense vector representing a position, is added to each token's embedding, the i-th encoding to the i-th token. A simple implementation is a trainable matrix with one row per position, initialized with small random values, whose first rows are added to the inputs, with a little dropout. Broadcasting adds the same encodings to every sentence in the batch.

The original paper proposed fixed encodings built from sine and cosine functions, but they perform no better than trainable ones, except perhaps on small transformers, so they are rarely used. Newer methods, such as relative position bias, rotary positional encoding, known as RoPE, and attention with linear biases, known as ALiBi, generally perform better.

---

Scaled dot-product attention.

Multi-head attention is built on scaled dot-product attention. In words: attention multiplies the query matrix by the transpose of the key matrix, divides by the square root of the key dimension, applies a softmax to each row, and uses the resulting weights to average the rows of the value matrix.

The queries and keys have the same number of dimensions, and the values have as many rows as the keys. The score matrix has one row per query token and one column per key token, a similarity for every pair, so its size grows with the product of the two lengths. Sequences cannot be too long. This is the critical quadratic context window problem. After the softmax, each row sums to one, and each output row is a weighted sum of the values, favoring those whose keys align best with that query. Dividing by the square root of the key dimension keeps the scores from saturating the softmax, which would produce tiny gradients, and it speeds up and stabilizes training.

Keys can be masked out by adding negative infinity to their scores before the softmax, which gives them zero weight. This hides padding tokens, and future tokens in the decoder. PyTorch's scaled dot product attention function implements this and accepts extra leading dimensions, such as the batch and the heads, so it computes everything at once.

---

Multi-head attention.

A multi-head attention layer is a set of scaled dot-product attention layers, called heads, each preceded by linear projections of the queries, keys, and values. The heads' outputs are concatenated and pass through a final linear projection.

A token's representation encodes many characteristics at once. For "like", these include its meaning, its being a verb, its present tense, and its position. A single attention layer could only query them all together. With several heads, the first projections can give each head a different subspace, for example one where only "verb in the present tense" remains, while another head focuses on meaning. The heads perform the lookups, and the final linear layer reorganizes the results. The model chooses its own representations, which makes it somewhat of a black box. Understanding what transformers do inside is an active research area called model interpretability.

In an implementation, the embedding size must be divisible by the number of heads. After the projections, the last dimension is split into heads, giving tensors of shape batch size by number of heads by sequence length by head dimension. Since matrix multiplication works on the last two dimensions, the scores and outputs are computed for every sentence and every head in one shot. Reshaping the result back concatenates the heads before the output projection.

Two optional boolean masks mark what to ignore. An attention mask, with one row per query token and one column per key token, controls which keys each query may see. A key padding mask marks the padding tokens in each sentence. Masked scores are set to negative infinity before the softmax. Zeroing the weights afterward would leave weights that no longer sum to one.

PyTorch's multi-head attention module works the same way but is much more optimized. Its batch-first option is off by default. Turning off its need weights option allows more optimizations, and so does setting its is causal flag when the attention mask is causal. A causal mask allows each token to attend to itself and earlier tokens only: it is true above the main diagonal and false elsewhere.

---

The encoder and decoder blocks.

An encoder block applies self-attention, adds the result to its input through a skip connection, and applies layer normalization. Then it applies the feedforward module, with another skip connection and normalization, plus some dropout. The feedforward module expands each token to two thousand forty-eight dimensions by default, applies ReLU, and projects back down. This reverse bottleneck makes the nonlinearity more expressive. The MobileNetV2 paper later called the idea an inverted residual network. In the encoder, the attention mask is usually unused, but the padding mask must be set.

A decoder block has three sublayers, each with a skip connection and layer normalization: masked self-attention, cross-attention over the encoder's output, called the memory, and the feedforward module. The padding masks must be set, and the target attention mask must be causal.

PyTorch provides ready-made encoder layer and decoder layer modules, with a batch-first option, is causal flags, and an activation that defaults to ReLU, though many state-of-the-art transformers use GELU. It also provides a transformer encoder that stacks copies of a layer, a matching decoder, and a full transformer module combining both. Adding a final linear layer and the cross-entropy loss completes the architecture.

---

An English-to-Spanish transformer.

The translation model embeds the English and Spanish token IDs, adds positional embeddings, passes both to PyTorch's transformer module, and applies a final linear layer. The tokenizer's attention masks hold ones for real tokens, but PyTorch expects true for tokens to ignore, so they are inverted. The causal mask keeps only the entries above the main diagonal of a square matrix of true values. The source padding mask is used in the encoder's self-attention and the decoder's cross-attention, and the target padding and causal masks in the decoder's self-attention.

Shrunk to four heads, two layers on each side, and one-hundred-twenty-eight-dimensional embeddings, and trained like the recurrent model for twenty epochs, this tiny transformer correctly translates "I like to play soccer with my friends at the beach". Between experiments, GPU memory should be freed by deleting unused objects, running Python's garbage collector, and emptying PyTorch's GPU cache.

---

Encoder-only transformers and BERT.

BERT, released by Google in 2018, showed that an encoder-only transformer can handle many language tasks, and it confirmed the power of self-supervised pretraining on a large corpus: fine-tuning on a small dataset is enough for excellent results.

Encoder-only models are generally not used for text generation, because they are much slower at it. Their attention is unmasked, so they are bidirectional, which is the B in BERT, and every new token forces everything to be recomputed. Decoders are causal, so they can cache and reuse earlier computations.

BERT is almost the original encoder, with three differences. It is much bigger: BERT-base has twelve blocks, twelve heads, and seven-hundred-sixty-eight-dimensional embeddings, and BERT-large has twenty-four blocks, sixteen heads, and one thousand twenty-four dimensions. It uses trainable positional embeddings for up to five hundred twelve tokens. It applies layer normalization before each sublayer rather than after each skip connection. This is called pre-LN, as opposed to post-LN, and it stabilizes training and reduces sensitivity to weight initialization. PyTorch's modules default to post-LN, with an option to switch. And the input can hold two segments, for tasks on sentence pairs. Each sentence ends with a separator token, and a trainable segment embedding is added to every token of each segment, along with the usual positional encodings.

---

BERT pretraining.

BERT was pretrained on English Wikipedia and BooksCorpus with two self-supervised tasks at once.

The masked language model task selects each token with fifteen percent probability, and the model must predict the original selected tokens. This is a cloze task, like filling in blanks: "She had fun at the birthday party" might become "She mask fun at the mask party", with the loss computed only on the selected positions. Most selected tokens become the mask token, but ten percent are replaced by random tokens, which forces the model to work without mask tokens, as downstream tasks have none. Another ten percent are left unchanged, which encourages the model to attend to the token at the predicted position instead of learning to ignore it.

Next sentence prediction asks whether two sentences are consecutive: "The dog sleeps" and "It snores loudly" are, while "The dog sleeps" and "The Earth orbits the Sun" are not. A class token, CLS, is inserted at position zero, and its output goes through a binary classification head. The aim was to make that output a good representation of the whole input, but averaging all the token embeddings later proved better, and next sentence prediction did not help much, so most later models dropped it.

The Transformers library can pretrain a BERT from scratch, for example on a domain-specific corpus. A configuration sets the model size, a masked language model is created with random weights, and a dataset such as WikiText is tokenized. A data collator for language modeling builds the batches and applies the fifteen percent masking rules, and the Trainer runs the training. After one epoch, asked to fill in "The capital of blank is Rome", the model suggests a comma. The BERT authors trained for about four days on sixteen TPU devices and far more data. It is usually better to start from a model pretrained on a similar corpus, optionally continue masked language model training, then fine-tune.

---

Fine-tuning BERT.

For sentence classification, such as sentiment analysis, only the class token's output is used, and a new classification head replaces the next sentence prediction head. The model is fine-tuned with cross-entropy, optionally with lower learning rates for lower layers, or with BERT frozen for the first epochs. This gave excellent results on CoLA, which asks whether sentences are grammatical, and it works even with small datasets. Adding the masked language model loss to the fine-tuning loss stabilizes training and reduces overfitting.

For token classification, the head is applied to every token. Named entity recognition tags names, dates, places, and organizations, which is common in legal, financial, and medical applications. Other examples are tagging grammatical errors, token-level sentiment, parts of speech, and dialogue acts such as questions and greetings.

Sentence pairs work like single sentences with two segments, for natural language inference, paraphrase detection, or deciding whether a sentence answers a question. For multiple choice question answering, BERT runs once per answer, with the question in segment zero and the answer in segment one. A one-unit linear layer on the class token scores each answer, and a softmax compares the scores.

For extractive question answering, the question goes in segment zero and a context text in segment one, and BERT must locate the answer in the context. A two-unit linear layer gives each token a start score and an end score, trained as two binary tasks. At inference time, the model picks the start and end positions with the highest combined score, with the start no later than the end and the answer not too long. This beat the state of the art on the SQuAD question answering dataset.

---

Sentence embeddings.

BERT can score the similarity of two sentences, but finding the most similar pair among N sentences would mean running it on about N squared pairs, which can take hours. Sentence-BERT, or SBERT, is fine-tuned to produce good sentence embeddings. Each sentence is embedded once, and pairs are compared with cosine similarity, the cosine of the angle between two vectors, from minus one for opposite to one for perfectly aligned. That takes seconds.

Sentence embeddings power text clustering, with algorithms such as k-means or HDBSCAN, often after reducing dimensionality with PCA or UMAP. They power semantic search, which embeds documents ahead of time and returns those closest to the query's embedding, matching meaning rather than keywords. They can also rerank the results of an existing search system. Vector databases, such as Pinecone, Weaviate, ChromaDB, Qdrant, and Milvus, store and search embeddings.

The Sentence Transformers library, created by UKPLab and maintained by Hugging Face, makes SBERT models easy to use. Its small, fast all MiniLM model rates "She's shopping" closer to "She bought some shoes" than to "She's working".

---

RoBERTa and DistilBERT.

RoBERTa, from Facebook AI in 2019, is similar to BERT but better across the board, mostly because it was pretrained longer on more data. It dropped next sentence prediction and used dynamic masking: tokens are masked on the fly, so the same text is masked differently in each epoch, which adds diversity and reduces overfitting.

DistilBERT, from Hugging Face in 2019, is forty percent smaller and sixty percent faster than BERT, yet reaches about ninety-seven percent of its performance on most tasks. It suits mobile devices, low-latency applications, and quick fine-tuning. It was trained with distillation, introduced by Geoffrey Hinton and colleagues in 2015. A small student model learns to match the estimated probabilities of a larger teacher model. These soft targets give the student the right distribution directly, so training is faster and more data-efficient, and it often beats training the student from scratch.

During distillation, both models' logits are divided by a temperature above one, typically two, which smooths the probabilities and reveals more about the alternatives. Hinton called this dark knowledge. For "It's sunny and I feel blank", a teacher's seventy-two percent for "great", twenty-seven for "good", and half a percent for "bad" become about sixty, thirty-six, and five percent, showing that "bad" is plausible though unlikely. DistilBERT's loss also includes the masked language model loss and a cosine embedding loss that aligns the student's final hidden states with the teacher's, so the student learns to think like the teacher. TinyBERT later also aligned internal states such as attention weights.

---

ALBERT, ELECTRA, and DeBERTa.

ALBERT, from Google Research in 2019, shares the same weights across all encoder layers, so it is much smaller than BERT, though not faster. It suits memory-limited settings. It also factorizes the embeddings. BERT-large's embedding matrix, about thirty thousand tokens by one thousand twenty-four dimensions, has over thirty million parameters. ALBERT uses one-hundred-twenty-eight-dimensional embeddings followed by a linear layer projecting them up, which takes under four million. It also replaced next sentence prediction with sentence order prediction, deciding which of two consecutive sentences comes first. The task is harder and gave much better sentence embeddings.

ELECTRA, from Google Research in 2020, introduced replaced token detection. A small generator, trained with masked language modeling, fills each masked position with a token sampled from its top predictions. A larger discriminator, the final model, must say for each token whether it is original or replaced. "She likes him" might become "She pushes him", and the discriminator should flag only the second token. As the generator improves, replacements become harder to spot. Because the discriminator learns from every token, not just the masked ones, training is more sample-efficient and converges faster, generally matching larger BERT models, though the extra complexity is not always worth it.

DeBERTa, from Microsoft in 2021, beat the state of the art on many NLU tasks. It has no positional embedding layer. Instead, each attention layer uses learned embeddings of the relative position between query token and key token when computing scores, in a form called disentangled attention that combines meaning and position flexibly. DeBERTaV3 added ELECTRA-style replaced token detection and remains popular for NLU, but disentangled attention adds cost, so later models chose simpler approaches.

---

Encoder models on the Hugging Face Hub.

The Hub has encoder variants of many sizes, for other languages, such as CamemBERT for French and IndicBERT for twelve Indian languages, cased and uncased, and tuned for specific tasks. There are domain-specific models, such as ClinicalBERT, SciBERT, PubMedBERT for biomedicine, FinBERT for finance, GraphCodeBERT for code, and LexLM for law. GraphCodeBERT was also pretrained to find where variables are defined and used, and to predict data flow. Compressed models, trained with distillation, weight sharing, or quantization, include DistilBERT, TinyBERT, MobileBERT, and MiniLM. Adapter models keep a standard model frozen and train only small added components called adapters, which makes fine-tuning faster and cheaper and works with little data. Large organizations now focus on decoders, but encoders remain fast, accessible, and immensely useful.

---

Decoder-only transformers and GPT-1.

Alec Radford and other OpenAI researchers built the first decoder-only model, GPT, released in June 2018 and now called GPT-1. The name stands for Generative Pre-Training. It was pretrained on about seven thousand books to predict the next token, so it generates text one token at a time: given "Happy birthday", it predicts "to" after "birthday", "to" is appended, and the process repeats.

Decoder-only models excel at text generation: autocompletion, code, free-form question answering, some math and logical reasoning, and chatbots. They can summarize and translate, but encoder-decoder models remain popular there, since the encoder understands the source better. They can classify, but encoder-only models are faster and often as good with fewer parameters. A decoder is autoregressive, so it runs once per generated token, though caching helps a lot.

GPT-1 was pretrained on batches of sixty-four sequences of exactly five hundred twelve tokens, predicting the next token at every position, with no padding and no special tokens. That is simpler than BERT's pretraining, and every position gets equal data. Each block has only masked self-attention and a feedforward network, with no cross-attention, since there is no encoder. It has twelve layers, seven-hundred-sixty-eight-dimensional embeddings, and twelve heads, for one hundred seventeen million parameters. PyTorch's transformer decoder module cannot build it, because its cross-attention cannot easily be removed, so the transformer encoder module with a causal mask is used instead.

GPT-1's generated text, such as a story about English-speaking unicorns found in an unexplored valley, seemed human-written at the time. Fine-tuned for entailment, semantic similarity, reading comprehension, and common sense reasoning, it beat the state of the art on many tasks, with small changes: a classification head on the last token's output, two sentences joined by a delimiter token, a plain dollar sign, and for similarity, both sentence orders processed and their outputs added.

---

GPT-2 and zero-shot learning.

GPT-2, published in February 2019 by Alec Radford, Jeffrey Wu, and other OpenAI researchers, is a similar but larger architecture in four sizes. The largest has forty-eight layers, twenty heads, sixteen-hundred-dimensional embeddings, and a context window of one thousand twenty-four tokens, for over one and a half billion parameters. Its training data, WebText, is about eight million web pages linked from highly ranked Reddit pages, since Common Crawl contained too much gibberish.

GPT-2 performed well on many tasks without fine-tuning, which is called zero-shot learning. Ending a question with "A" and a colon, as in "What is the capital of New Zealand?", makes it answer "Wellington". Appending "TL DR", short for too long, didn't read, often yields a decent summary. A prompt with a few French and English example pairs, followed by a French sentence, gets completed with its translation. Zero-shot performance grew steadily with size: each doubling gave a roughly constant improvement, a log-linear relationship.

OpenAI initially withheld the largest GPT-2, citing risks such as automated disinformation and spam, while skeptics saw a publicity stunt and a shift toward closed AI. It was released months later, and it was OpenAI's last open model until the GPT-OSS open-weight models of August 2025.

---

GPT-3 and in-context learning.

Following a bigger-is-better philosophy, OpenAI created GPT-3 in 2020, trained on about five hundred seventy gigabytes of text. It was far better than GPT-2, especially at zero-shot tasks, and it generalized remarkably well from a few examples. This is few-shot learning, or one-shot learning with a single example. The examples simply go in the prompt, which is called in-context learning. After one example sentence about Alice visiting her friend, with the blank filled in as "Bob", a sentence about George buying a ball, a glove, and a blank gets completed with "bat". In-context learning is new, but one-shot and few-shot learning are old ideas.

---

Generating text with GPT-2.

The Transformers library's auto model for causal language modeling loads GPT-2, by default the small version with one hundred twenty-four million parameters. A causal language model can generate text. The automatic device map places the model on the best device, and can split a large model across GPUs. The automatic data type picks sixteen-bit floats when the hardware supports them, which halves memory use and speeds things up.

A small wrapper tokenizes a prompt, moves it to the GPU, calls the model's generate method to add up to fifty tokens, and decodes the result. GPT-2 was pretrained without padding, so the end-of-sequence token serves as the padding token. Decoder-only models often pad on the left, since new tokens are added on the right.

Asked to continue "Scientists found a talking unicorn today. Here's the full story", GPT-2 starts well, then repeats one sentence about a field in New Mexico. By default, the generate method is greedy. That suits structured output and question answering, but creative writing gets stuck in loops. Enabling sampling makes the text less repetitive. Other key arguments are the temperature, top k, top p, and the number of beams for beam search.

Top-p sampling, or nucleus sampling, is often preferred over top-k because it adapts to the distribution. After "The capital city of France is", only "Paris" is likely, and top-p always picks it, while top-k may pick a wrong token. After "My favorite city is", many tokens are likely, and top-p can pick any of them, while top-k considers only a few. With top p set to zero point six, the unicorn story improves greatly.

---

Question answering with GPT-2.

A prompt template turns GPT-2 into a capital city finder: one solved example, "Capital city of France equals Paris", then the same phrase for the requested country. Including an example of the task and format is in-context learning. The function generates ten tokens and keeps the first line of the answer. It returns London for the United Kingdom, even when given "England" or a misspelled "Big Britane", and Mexico City for Mexico.

It is far from perfect. It answers Toronto for Canada, a common misconception picked up from the web. When unsure, it just repeats the country's name, which happens about thirty percent of the time, perhaps because some capitals share their country's name, such as Luxembourg and Singapore. For inputs that are not countries, it often answers "Paris", its only example. GPT-2 XL, with one and a half billion parameters, does clearly better.

---

Running a larger model: Mistral-7B.

Mistral-7B, from the French startup Mistral AI, has seven billion parameters and uses grouped-query attention and sliding-window attention for speed and performance. It is released under the permissive Apache 2.0 license and fits on Colab GPUs.

It is gated on the Hugging Face Hub: users must log in and agree to share their identity with the authors, which helps authors monitor usage, reduce abuse, and contact users. Downloading it takes an account, accepting the terms on the model page, and an access token. In production, a token should have minimal permissions, such as read access to just this model, in case it leaks. Tokens should be kept in a password manager and refreshed if they may be compromised. A token should never be hardcoded in a notebook. Colab's secrets manager stores it on Google's servers, and each notebook must be explicitly granted access, only if its author is trusted.

After logging in to the Hub with the token, Mistral-7B loads exactly like GPT-2. It finds the capital of almost every country, and its few mistakes are reasonable, such as Rome for the Vatican and Monte Carlo for Monaco.

---

Turning a language model into a chatbot.

A base model is not a chatbot. Given "List some places I should visit in Paris", Mistral-7B does not answer. It just continues the text, writing as someone planning a trip to Paris.

Prompt engineering is the art of tweaking a prompt until the model reliably behaves as intended. Adding an introduction, "Bob is an amazing chatbot. It knows everything and it's incredibly helpful", and labeling turns with "Me" and "Bob", called role tags, makes the model answer as Bob: the Eiffel Tower, the Louvre, and the Arc de Triomphe. It then invents the rest of the conversation, so everything after Bob's first answer is cut off.

For a multiturn conversation, the context is kept and each new prompt is appended with the role tags, so Bob knows what "the first place" means. A small chatbot class does this in about twenty lines, generating in chunks until a new "Me" turn appears or the answer gets too long. Bob can then describe the Eiffel Tower and suggest the Colosseum for Rome.

Bob has problems. Asked for five jokes, it tells the same joke five times. Asked how to make cookies, it just says to mix flour, sugar, butter, and eggs. Asked how to prepare a bank robbery, it suggests a mask and a gun.

---

Prompt engineering techniques.

Answers vary widely with phrasing, especially for large language models, or LLMs. Useful levers include word choice, added context, examples, a persona to imitate, such as "You are a friendly real-estate expert", the output format, and pitfalls to avoid. Prompt variants can be tested automatically, and automatic prompt optimization techniques such as prompt tuning prepend a few learned embeddings to the inputs.

Prompt chaining splits a task into subtasks, each output feeding the next prompt. Instead of asking for a lesson on fluid dynamics directly, the LLM writes an outline, checks it, then writes the lesson from it, which often gives much better results.

Chain-of-thought prompting asks the LLM to reason step by step, explicitly or through examples, which improves reliability on reasoning tasks. Chain-of-thought with self-consistency runs it several times and keeps the most frequent answer. Tree-of-thoughts generates several options, called thoughts, at each step, evaluates them, explores the most promising, and backtracks from dead ends. It reasons very well but is costly. Multi-agent debate, where several LLMs argue, is also costly. A lighter option asks one LLM to critique and refine its own answer.

LLMs tend to make up convincing false statements, called hallucinations. A key countermeasure is adding relevant, reliable context to the prompt on the fly. A medical chatbot asked about aspirin can retrieve facts from a medical database and include them. This is retrieval augmented generation, or RAG. An LLM can also write detailed prompts for other models, such as image generators.

Prompting alone is probably not enough to make Bob reliably safe. A user could jailbreak it, tricking it into ignoring or revealing its directives, or attempt targeted data extraction of personal information that leaked into the training data. Fine-tuning makes the model more helpful and safer.

---

Fine-tuning for chat with SFT and RLHF.

A chatbot starts with pretraining a transformer, usually decoder-only, on a huge corpus with next token prediction. This is the most costly step, and it yields a base model, such as Mistral-7B or GPT-3. Fine-tuning it to be more conversational gives a conversational or dialogue model, and fine-tuning it to follow instructions gives an instruct model. Chatbot models usually get both, as with Mistral-7B-Instruct. Strictly, a base model has only been pretrained, while a foundation model is any model adaptable to many tasks, but the terms are often used interchangeably.

Chat fine-tuning has two steps. Supervised fine-tuning, or SFT, trains on a curated dataset of conversations, question and answer pairs, code, math solutions, role-play, and safe refusals, such as declining to explain a bank robbery. It is ordinary next token prediction, usually computing the loss only on answer tokens. This is called loss masking, and it focuses the model on its answers rather than on imitating user prompts. Then human evaluators rank the model's responses, and the model is fine-tuned to produce higher-ranked ones, with reinforcement learning from human feedback, or RLHF, or with direct preference optimization, or DPO.

OpenAI introduced this recipe in January 2022 with InstructGPT, based on GPT-3. RLHF, from a 2017 paper by OpenAI and DeepMind researchers, trains a reward model to predict human preferences, then fine-tunes the LLM with a reinforcement learning algorithm called proximal policy optimization to favor high-reward answers. The LLM is kept from drifting far from the original model. Otherwise it could overfit the preferences and forget useful pretrained behavior, which is called reward hacking. RLHF works and is widely used, but it can be unstable and tricky.

---

Direct preference optimization.

DPO, proposed by Stanford researchers in May 2023, often matches or beats RLHF and is simpler, more stable, and more data-efficient. Each sample has a prompt and two answers, one chosen by human raters and one rejected. The goal is to make the chosen answer more likely than the rejected one without drifting far from a frozen reference model, usually the model right after SFT. This is contrastive learning, learning by comparing positive and negative examples. It is roughly equivalent to RLHF but needs no reward model and no reinforcement learning.

The DPO loss works like this. For each answer, take the model's log probability of the answer given the prompt, minus the reference model's. Subtract the rejected answer's difference from the chosen answer's, and multiply by a hyperparameter called beta. The loss is the negative log of the sigmoid of the result. A high beta keeps the model close to the reference, and a low beta follows the preferences more. It is typically between zero point one and zero point five.

To get an answer's log probability, the prompt and answer are concatenated and run through the model, and a log softmax gives log probabilities for every possible next token. Only the actual next token's log probability matters, so it is picked out at each position. For "The capital of Argentina is", Mistral-7B gives "Buenos" about eleven percent and "Madrid" almost zero, and after "Buenos", "Aires" gets over ninety-nine percent. Summing the answer tokens' log probabilities gives the whole answer's log probability.

Finding where the answer starts is cumbersome in padded batches, so the log probability of the full sequence, prompt plus answer, is used instead. It equals the prompt's log probability plus the answer's given the prompt, and the prompt's term cancels out between chosen and rejected. So the log probabilities are masked to ignore padding and summed per sequence. The DPO loss computes these sums for the chosen and rejected inputs with the trained model, and again with the frozen reference model, then applies the formula.

---

Fine-tuning with the TRL library.

Hugging Face's transformer reinforcement learning library, TRL, implements SFT, RLHF, and DPO with an interface like the Trainer's. For SFT, the Alpaca dataset has about fifty-two thousand instructions and responses generated by an OpenAI model, such as "What are the three primary colors?" answered with "red, blue, and yellow". It teaches single instructions, and a multiturn dataset such as OpenAssistant's would come next to teach conversations and role tags. There is no standard for role tags. Many models use "User" and "Assistant", OpenAI's ChatML format uses special tags for the user, the assistant, and the system, and Anthropic uses "Human" and "Assistant". Here, Alpaca is rewritten with Anthropic-style tags, and an SFT trainer fine-tunes GPT-2, which is too small to learn much but shows the process.

For DPO, Anthropic's HH-RLHF dataset targets helpful and harmless chatbots. Asked whether stealing from a convenience store is better in the morning or at night, its chosen answer declines, and its rejected answer discusses foot traffic and cameras. A DPO trainer fine-tunes the SFT model on it.

That completes the path from a transformer to a chatbot model. Alternatively, an already fine-tuned model such as Mistral-7B-Instruct can be downloaded. In the same chatbot class, it tells five different jokes, gives a clear cookie recipe, and declines to help rob a bank.

---

From a chatbot model to a chatbot system.

The last step is deploying the model in a full system with a web interface, an app, and possibly an API. Chatbots increasingly rely on tools, coordinated by a component called the orchestrator.

Calculator. The orchestrator can detect a math expression, evaluate it with a calculator, and add the result to the prompt, so the model only phrases the answer. Alternatively, the model can be fine-tuned to invoke tools itself, which is called tool augmentation or function calling. It outputs a special calculator section, and the orchestrator evaluates it and either substitutes the result or calls the model again with it, which costs more but allows the model to comment on the result.

Web search. The orchestrator can fetch a web page and inject its text, or only a summary or the chunks most similar to the prompt. The model can also request searches: asked for the population of Canada's capital, it can search for the population of Ottawa, and the orchestrator feeds back the results and their sources, so the answer is current and cited.

Retrieval augmented generation extends this to any data source, including private ones such as a company's SQL database, PDF documents, or knowledge bases. A support chatbot told that a new fridge is humming can search the internal knowledge base, for example with a vector database, and answer from the relevant chunks. Retrieval can also flag unsafe topics so the chatbot politely declines.

Memory, or persistent context, stores user facts across conversations, such as a preferred name. The orchestrator saves them in a database and injects them into later conversations, either all at once or by retrieving only the relevant ones, so memory does not crowd the context window.

Agentic behavior. A model fine-tuned to plan and carry out multistep tasks with tools is called an agent. For a deep search, it may ask clarifying questions, plan, search, analyze, and repeat until it can write a report. A model fine-tuned to reason without tools is a reasoning model.

Other tools include unit conversion, weather, document upload, code interpreters, and Wolfram Alpha. Not nuclear missiles, though. In 1983, Soviet lieutenant colonel Stanislav Petrov arguably prevented a nuclear war by judging a missile alert to be a false alarm. LLMs are often unreliable, so humans should stay in the loop for important decisions.

---

The Model Context Protocol.

The Model Context Protocol, or MCP, is an open standard proposed by Anthropic for how an AI system communicates with MCP servers that provide tools and resources, such as files, email, calendars, and weather. MCP says nothing about the LLM. The orchestrator must detect when the LLM wants a tool, for example when it outputs a small JSON message asking for the weather in Paris. The orchestrator then sends an MCP request to a weather server and feeds the response back to the LLM, which answers that it will be sunny with a high of twenty-three degrees Celsius. The LLM can also output MCP requests directly, for the orchestrator to validate and route.

MCP does more than a plain web API. Connections are long-lived, allowing fast, stateful, two-way communication. The components managing them are MCP clients, and the software hosting them, usually the orchestrator, is the MCP host. MCP also has an AI-friendly discovery mechanism: a client can ask a server for a text description of what it does and how to use it, including its functions and parameters, making it a self-documenting API for AIs. Servers can also ask clients about their capabilities, such as displaying images. Once an LLM knows which servers exist, it can work out how to use them, so connecting a new server takes little more than configuration.

---

Structured generation.

When generating JSON, a model can make syntax errors, such as an extra bracket, or schema errors, such as a missing field or a string instead of an integer. Post-processing fixes are unreliable. Structured generation constrains sampling to valid tokens only, picking the valid token the model prefers most. In the Transformers library, a custom logits processor passed to the generate method can set the logits of invalid tokens to negative infinity. Libraries such as Outlines and Guidance make this easier.

---

Chatbot libraries and tools.

LangChain builds LLM applications by chaining components such as prompt templates, models, memory, and tools. LangGraph, built on it, targets long-running, stateful, agentic workflows. Smolagents is Hugging Face's library for agents. Haystack builds question answering systems, typically with RAG. LlamaIndex ingests, indexes, and queries data such as PDF files and databases.

For chatting with models locally, LM Studio is a desktop app, Ollama is a command-line tool that can also serve models through an API, and Text Generation Web UI is a feature-rich web interface with plug-ins. LM Studio and Ollama run models on llama cpp, a highly optimized C plus plus library. These can be combined, for example LangChain orchestrating a local model served by Ollama, with Haystack retrieving from a vector database.

---

Encoder-decoder models.

Encoder-decoder models remain very relevant for translation and summarization, since the bidirectional encoder gives the decoder excellent contextual embeddings of the source, usually beating a decoder-only model of similar size.

Google's T5, from 2019, was the first to frame every NLP task as text to text. A prefix such as "translate English to Spanish", "summarize", or "classify" selects the task, and the model outputs text: the translation, the summary, or the class name. For zero-shot classification, the classes are listed in the prompt. T5 was pretrained with masked span corruption, which masks one or more contiguous spans. Its variants include mT5, a multilingual version covering over one hundred languages, ByT5, which works on bytes with no tokenizer but did not catch on, FLAN-T5, which is instruction-tuned with excellent zero-shot and few-shot ability, and UL2 and FLAN-UL2, pretrained with several objectives.

Meta's BART, from 2020, was pretrained to restore corrupted text, with masked, deleted, or inserted tokens and shuffled sentences. It is particularly good at text generation and summarization, and mBART is its multilingual variant. Encoder-decoders are also common in vision tasks with several outputs, such as object detection and segmentation, and in multimodal models.

---

Summary.

The Transformer is an encoder-decoder built from dense layers and multi-head attention, with no recurrence, so it trains in parallel, scales well, and captures long-range patterns. Positional encodings supply word order. Attention takes the softmax of the dot products between each query and every key, divided by the square root of the key dimension, and uses those weights to average the values. Multi-head attention runs several heads on different learned projections. Masks hide padding and, in the decoder, future tokens.

Encoder-only models such as BERT are pretrained with masked language modeling and fine-tuned for classification, token tagging, sentence pairs, and extractive question answering. SBERT produces sentence embeddings for similarity, search, and clustering. RoBERTa, DistilBERT, ALBERT, ELECTRA, and DeBERTa improved pretraining, size, and efficiency.

Decoder-only models such as GPT are pretrained with next token prediction and generate one token at a time. Larger models can learn new tasks from instructions or a few examples in the prompt. Sampling, temperature, top-k, top-p, and beam search control generation.

A chatbot starts from a base model, is fine-tuned with SFT and then RLHF or DPO to be helpful and safe, and runs in a system whose orchestrator adds tools such as calculators, web search, RAG, and memory, possibly through MCP. Prompt engineering techniques such as chain-of-thought improve answers. Encoder-decoder models such as T5 and BART remain strong for translation and summarization.
