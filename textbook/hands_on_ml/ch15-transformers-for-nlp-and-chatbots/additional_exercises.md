# Chapter 15 — Transformers for Natural Language Processing and Chatbots

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's end-of-chapter questions are in
exercises.md, with answers written by Claude, since the author hasn't
published any; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Walk through the forward pass of a from-scratch multi-head attention module, giving the tensor shape at each step.

Each head computes scaled dot-product attention, softmax(QKᵀ / √d)·V, and all heads run at once as batched matrix products (h heads of size d = embed_dim / h, batch size B):

1. Project the query, key and value with three Linear layers.
2. split_heads() views each [B, L, h·d] tensor as [B, L, h, d] and transposes it to [B, h, L, d]. The @ operator only multiplies the last two dimensions, so B and h act as batch dimensions.
3. scores = q @ k.transpose(2, 3) / √d, shape [B, h, Lq, Lk]. Dividing by √d keeps the softmax from saturating, which would shrink gradients. Masked positions are set to −∞.
4. Softmax over the last dimension makes each row sum to 1 (−∞ gives 0); weights @ v gives [B, h, Lq, d].
5. Transpose to [B, Lq, h, d] and reshape to [B, Lq, h·d], concatenating the heads, then apply the output Linear layer.

## 2. Which masks does nn.Transformer need when you train a translation model, and how do you build them?

Four boolean masks, in which True means "don't attend to this key":
- src_key_padding_mask [B, Ls]: hides source padding from the encoder's self-attention.
- memory_key_padding_mask [B, Ls]: the same mask, hiding source padding from the decoder's cross-attention over the encoder outputs.
- tgt_key_padding_mask [B, Lt]: hides target padding.
- tgt_mask [Lt, Lt]: the causal mask for the decoder's masked self-attention, True above the main diagonal so each position sees only itself and earlier tokens.

Tokenizers return attention masks with 1 for real tokens, so invert them with ~mask.bool(). Build the causal mask with torch.triu(torch.full((L, L), True), diagonal=1) or nn.Transformer.generate_square_subsequent_mask(L, dtype=torch.bool). Create the module with batch_first=True if your tensors are [B, L, E], since the default is False.

The causal mask is also how you get a GPT-style model: call nn.TransformerEncoder with a causal mask. nn.TransformerDecoder won't do, because its cross-attention layers can't easily be removed.

## 3. How do you adapt a pretrained BERT model to sentence classification, token classification, multiple-choice questions and extractive question answering?

Add a small head on the encoder's outputs and fine-tune:
- Sentence classification (e.g., sentiment): a new classification head on the [CLS] token's output. Sentence-pair tasks such as natural language inference work the same way, with the sentences separated by [SEP] and told apart by segment embeddings.
- Token classification (e.g., named entity recognition): a classification head applied to every token's output.
- Multiple choice: run BERT once per candidate answer (question in segment 0, answer in segment 1). A one-unit linear layer turns each [CLS] output into a score, and cross-entropy over the candidates' scores trains it.
- Extractive QA: with the question in segment 0 and the context in segment 1, a two-unit linear layer outputs a start score and an end score for every token. The predicted answer is the span i ≤ j that maximizes startᵢ + endⱼ, up to a maximum answer length.

## 4. Why is a sentence-embedding model like SBERT better than plain BERT for finding similar sentences, and how do you use one?

BERT can be fine-tuned to score how similar two sentences are, but it must read each pair together, so finding the most similar pair among N sentences takes O(N²) forward passes, which can take hours. SBERT is a BERT variant fine-tuned to produce good sentence embeddings: encode each sentence once, then compare embeddings with a cheap measure such as cosine similarity (from −1 to +1). That takes seconds. With the Sentence Transformers library:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # small and fast
embeddings = model.encode(sentences, convert_to_tensor=True)
similarities = model.similarity(embeddings, embeddings)  # [N, N] matrix
```

The same embeddings power semantic search (embed the documents once, embed each query, return the nearest documents, often from a vector database), text clustering (e.g., k-means or HDBSCAN on the embeddings), and reranking an existing search engine's results.

## 5. How do you load a pretrained causal language model with the Transformers library and generate text from a prompt?

Load the tokenizer and the model, then wrap tokenization, generate() and decoding in a helper:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto", dtype="auto")

def generate(prompt, max_new_tokens=50, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

- AutoModelForCausalLM picks the right class for the checkpoint (GPT2LMHeadModel here).
- device_map="auto" places the model on the best available device, even sharding it across GPUs if it's too big for one.
- dtype="auto" picks the weight type from the checkpoint and your hardware, typically 16-bit floats on a modern GPU: half the memory, and faster.
- GPT-2 was pretrained without a padding token, so the end-of-sequence token is reused for padding. Decoder-only models are often padded on the left, since new tokens get appended on the right.
- The output starts with the prompt, so slice it off if you only want the continuation.

## 6. State the DPO loss and explain its terms. Why is DPO often preferred to RLHF?

For a prompt x with a human-preferred (chosen) answer y_c and a rejected answer y_r:

J(θ) = −log σ(β·[δ(y_c) − δ(y_r)]), where δ(y) = log p_θ(y | x) − log p_ref(y | x)

- p_θ is the model being fine-tuned; p_ref is a frozen reference model, usually the SFT model you started from.
- δ(y) measures how much more likely the model makes answer y than the reference does.
- σ is the sigmoid, so the loss falls as the chosen answer gains probability relative to the rejected one.
- β (typically 0.1 to 0.5) sets the sigmoid's steepness: a high β keeps the model close to the reference, a low β follows the preferences more.

It's roughly equivalent to RLHF but needs no separate reward model and no reinforcement learning (PPO), so it's simpler, more stable and more data efficient. Compute log σ with F.logsigmoid() for numerical stability.

## 7. How do you compute the log-probability that a causal language model assigns to each sequence in a padded batch?

The logits at position t predict token t + 1, so align logits[:, :-1] with the targets input_ids[:, 1:]. You could apply F.log_softmax() and pick each target's value with torch.gather(), but F.cross_entropy() does both in one step:

```python
logits = model(**encodings).logits  # [B, L, vocab_size]
targets = encodings.input_ids[:, 1:]  # the next token at each position
token_log_probas = -F.cross_entropy(
    logits[:, :-1].permute(0, 2, 1), targets, reduction="none")  # [B, L-1]
mask = encodings.attention_mask
valid = mask[:, :-1] * mask[:, 1:]  # skip pairs involving padding
seq_log_probas = (token_log_probas * valid).sum(dim=1)  # [B]
```

- permute(0, 2, 1) because cross_entropy expects the class dimension at index 1.
- reduction="none" keeps one value per token instead of the mean.
- The minus sign turns the negative log-likelihood back into a log-probability.
- Summing gives the log-probability of the whole sequence. For DPO you can score prompt plus answer instead of the answer alone, since log p(xy) = log p(x) + log p(y | x) and the log p(x) terms cancel between the chosen and rejected answers.

## 8. How can you guarantee that an LLM's output is valid JSON, or follows any other grammar?

Post-processing the output to fix syntax or schema errors (an extra bracket, a missing field, a string instead of an integer) isn't fully reliable. Structured generation constrains the sampling itself: at each step, work out which tokens could legally come next given the text so far, and pick the one the model prefers among those only, even if it would rather output an illegal token. For example, if the schema says name is a string, then after {"name": the only legal continuations are a space or an opening quote.

With the Transformers library, subclass transformers.LogitsProcessor: its __call__(input_ids, scores) method receives the logits just before each token is chosen and can set the logits of all invalid tokens to −∞. Pass it to generate() through the logits_processor argument (in a LogitsProcessorList). Since this is low-level, libraries such as Outlines or Guidance handle the grammar and token bookkeeping for you.

## 9. What does it mean that T5 frames every NLP task as text-to-text, and how was it pretrained?

T5 is an encoder-decoder model whose inputs and outputs are always plain text, with a prefix in the input saying which task to perform: "translate English to Spanish: I like soccer" should produce "me gusta el fútbol", "summarize:" followed by a paragraph should produce its summary, and for classification the model writes the class name as text. One architecture, one training objective and one decoding procedure cover every task, which makes the model easy to pretrain on many tasks and just as easy to use. Listing candidate classes in the prompt even enables zero-shot classification.

T5 was pretrained with masked span corruption: like BERT's masked language modeling, except that whole contiguous spans of tokens are masked and the model must generate the missing text. BART, another encoder-decoder, uses a broader denoising objective (masked, deleted or inserted tokens, shuffled sentences), and it's particularly effective for text generation and summarization.
