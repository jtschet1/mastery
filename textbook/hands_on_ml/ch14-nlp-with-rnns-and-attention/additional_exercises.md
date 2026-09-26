# Chapter 14 — Natural Language Processing with RNNs and Attention

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. Walk through how the chapter builds its Shakespeare char-RNN, from raw text to per-character logits.

1. Lowercase the text and use the sorted set of its characters as the vocabulary (39 tokens), with char_to_id and id_to_char dictionaries to encode text as ID tensors and decode it back.
2. A CharDataset cuts the encoded text into 50-character windows; each target is the same window shifted one character ahead, so the model learns to predict the next character at every position. The DataLoader shuffles the windows.
3. The model chains nn.Embedding(39, 10), nn.GRU(10, 128, num_layers=2, batch_first=True, dropout=0.1) and nn.Linear(128, 39): a logit for every vocabulary character at every time step.
4. forward() permutes the logits from [batch, time, vocab] to [batch, vocab, time], because nn.CrossEntropyLoss and the Accuracy metric expect the class dimension second.
5. To predict the next character, take the last time step's logits, Y_logits[0, :, -1], and pick the argmax.

The window length caps the patterns it can learn at 50 characters.

## 2. Why feed a network embeddings rather than raw token IDs or one-hot vectors, and what does nn.Embedding compute?

Raw IDs mislead the model, which assumes numerically close inputs are similar, although tokens 12 and 13 may be unrelated. One-hot vectors avoid that, but they're as long as the vocabulary and mostly zeros: impractical for tens of thousands of words. An embedding represents each category as a small dense vector (say 300 dimensions instead of 50,000) that's trained with the rest of the model, so useful structure emerges, such as similar words ending up close together (representation learning).

nn.Embedding(num_embeddings, embedding_dim) holds a randomly initialized matrix with one row per category and simply looks rows up: an integer tensor of shape [batch, length] becomes a float tensor of shape [batch, length, embedding_dim]. Mathematically, that's one-hot encoding followed by nn.Linear(bias=False), without the wasted multiplications by zero. If you set padding_idx, that ID maps to a zero vector that receives no gradient updates.

## 3. How does the chapter generate text with its char-RNN, and what do temperature, top-k and top-p sampling control?

Greedy decoding (always appending the most likely next character) tends to repeat the same words. Instead, the chapter samples the next character from the model's predicted distribution with torch.multinomial(), appends it, and repeats:

```python
def next_char(model, text, temperature=1):
    X = encode_text(text).unsqueeze(dim=0).to(device)
    with torch.no_grad():
        Y_logits = model(X)  # [1, vocab, time]
    probas = F.softmax(Y_logits[0, :, -1] / temperature, dim=-1)
    return id_to_char[torch.multinomial(probas, num_samples=1).item()]
```

Dividing the logits by the temperature reshapes the distribution: near 0 it approaches greedy decoding (precise but repetitive), 1 keeps the model's own probabilities, and high values flatten it toward uniform, eventually producing gibberish. Use low temperatures for rigid text like equations and higher ones for creative text. Top-k sampling only samples among the k most likely tokens, and top-p (nucleus) sampling among the smallest set of top tokens whose total probability exceeds p; both cut off the unlikely tail.

## 4. How does byte pair encoding (BPE) build a subword vocabulary, and how do you train a BPE tokenizer with the Hugging Face Tokenizers library?

Subword tokens let a model handle rare or unseen words by composing familiar pieces (smartest = smart + est). BPE starts with the training text split into individual characters, then repeatedly adds the most frequent pair of adjacent tokens to the vocabulary as a new token, until the vocabulary reaches the desired size. Training one on the IMDb reviews with the Tokenizers library:

```python
import tokenizers

bpe_model = tokenizers.models.BPE(unk_token="<unk>")
bpe_tokenizer = tokenizers.Tokenizer(bpe_model)
bpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
bpe_trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=1000, special_tokens=["<pad>", "<unk>"])
bpe_tokenizer.train_from_iterator(train_reviews, bpe_trainer)
```

The Whitespace pre-tokenizer first splits the text into words and punctuation (dropping the spaces), so merges happen within those chunks, which speeds up training and gives cleaner tokens. The trainer puts the special tokens first, so padding is ID 0 and unknown is ID 1. Afterward, encode() returns an Encoding with .tokens, .ids, .offsets and .attention_mask, and decode() turns IDs back into text.

## 5. Compare byte-level BPE, WordPiece and Unigram LM tokenizers.

- Byte-level BPE (BBPE): BPE over UTF-8 bytes, via the ByteLevel pre-tokenizer, which also replaces spaces with a special character (Ġ) so decoding can restore them. With all 256 bytes in its vocabulary it never needs an unknown token, even for emojis. Fast, simple and great for multilingual text, but its splits can be awkward. Used by GPT models, Llama and RoBERTa.
- WordPiece: merges the pair with the highest score instead of the highest count, score(AB) ∝ freq(AB) / (freq(A) · freq(B)), which penalizes pairs of individually frequent tokens. Tokens inside a word get a ## prefix. It often yields shorter sequences than BPE. Used by BERT.
- Unigram LM: starts from a huge vocabulary and repeatedly drops the tokens whose removal least reduces the corpus likelihood, assuming tokens occur independently. Most meaningful tokens and shortest sequences, but slower; well suited to languages that don't separate words with spaces. Used by T5 and ALBERT.

## 6. How do you load and call a pretrained Hugging Face tokenizer, and what does it return?

Load it by checkpoint name with transformers.AutoTokenizer.from_pretrained(), then call it like a function on a list of strings:

```python
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
encoding = bert_tokenizer(reviews, padding=True, truncation=True,
                          max_length=500, return_tensors="pt")
ids, mask = encoding["input_ids"], encoding["attention_mask"]  # [batch, longest]
```

The result is a dictionary-like BatchEncoding. padding=True pads every text to the longest one in the batch, truncation=True with max_length cuts long texts, and return_tensors="pt" returns PyTorch tensors instead of lists of lists (which needs equal lengths, hence the padding). The attention mask holds 1 for real tokens and 0 for padding. Gotchas: GPT-2's tokenizer has no padding token, so it can't pad; BERT's tokenizer adds [CLS] (ID 101) at the start and [SEP] (ID 102) at the end unless you pass add_special_tokens=False. decode() maps IDs back to text, and transformers.PreTrainedTokenizerFast(tokenizer_object=...) gives a tokenizer you trained with the Tokenizers library this same API.

## 7. Walk through the chapter's GRU sentiment classifier for IMDb reviews, from the DataLoader to the loss.

1. The datasets hold raw text, so tokenization happens per batch in the DataLoader's collate_fn: it receives a list of samples, tokenizes the reviews with the pretrained BERT tokenizer (padding=True, truncation=True, max_length=200, return_tensors="pt"), and returns the BatchEncoding plus a float32 label tensor of shape [batch, 1].
2. forward() takes that BatchEncoding and embeds encodings["input_ids"] with nn.Embedding(vocab_size, 128, padding_idx=0), so padding tokens become fixed zero vectors.
3. A 2-layer nn.GRU (hidden size 64, batch_first=True, dropout=0.2) reads the sequence. It's a sequence-to-vector model, so only hidden_states[-1], the top layer's final state, is kept.
4. nn.Linear(64, 1) turns it into one logit per review (positive for a positive review), trained with nn.BCEWithLogitsLoss.

One flaw: the GRU still runs through any trailing padding and may forget the review by the end. Feeding it a packed sequence (lengths from the attention mask) makes it stop at each review's real end.

## 8. What does a bidirectional recurrent layer do, and what changes in the code when you set bidirectional=True on nn.GRU?

It runs two recurrent layers over the same inputs, one left to right and one right to left, and concatenates their outputs at each time step, so every position's representation also reflects what comes after it: to encode "right", you need to see whether "arm" or "to speak" follows. That suits text classification or an encoder, but not a decoder, which must stay causal.

Code changes:
- The outputs' last dimension doubles to 2 × hidden_size.
- The final hidden states have shape [2 × num_layers, batch, hidden_size], ordered layer 1 forward, layer 1 backward, layer 2 forward, and so on, so the top layer's two directions are hidden_states[-2:].
- The output layer's input size doubles, and the two top states must be concatenated per instance:

```python
top_states = hidden_states[-2:].permute(1, 0, 2).reshape(-1, 2 * hidden_dim)
return self.output(top_states)  # self.output = nn.Linear(2 * hidden_dim, 1)
```

## 9. Walk through the chapter's options for reusing pretrained parts in the IMDb classifier, from word embeddings up to BERT's contextualized outputs.

1. Pretrained word embeddings: copy BERT's embedding matrix into your own layer with nn.Embedding.from_pretrained(weights, freeze=True). Freezing stops large early gradients from wrecking them; unfreeze later to fine-tune. The limit: each word gets one vector whatever its context, so "right" is encoded the same in "left and right" and "right and wrong".
2. Contextualized embeddings, by reusing a whole pretrained language model (the idea behind ELMo and ULMFiT): run BERT and feed its per-token outputs to your GRU, which needn't be bidirectional since these embeddings already looked ahead.
3. Drop the GRU: during pretraining BERT learned to summarize the text in its first token, [CLS], so feed last_hidden_state[:, 0], or pooler_output (that vector passed through BERT's Linear + tanh pooler), to an nn.Linear head.

Freeze BERT at first with requires_grad_(False). The calls:

```python
bert = transformers.AutoModel.from_pretrained("bert-base-uncased")
out = bert(**encodings)  # encodings from the matching tokenizer
out.last_hidden_state    # [batch, length, 768]; "last" means last layer
```

## 10. How do you fine-tune BERT on IMDb with BertForSequenceClassification and the Trainer API?

BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) is BERT plus a new, untrained classification head. Hugging Face treats binary classification as 2-class classification: the model outputs two logits, so use cross-entropy and torch.softmax(), not BCE and sigmoid. If you also pass integer class labels, the output includes the loss: cross-entropy here (with num_labels=1 it would be MSE, for regression). The Trainer takes tokenized datasets, not data loaders:

```python
tok_train = imdb_train_set.map(tokenize_batch, batched=True)
tok_valid = imdb_valid_set.map(tokenize_batch, batched=True)
args = TrainingArguments(output_dir="my_imdb_model", num_train_epochs=2,
                         eval_strategy="epoch", save_strategy="epoch",
                         load_best_model_at_end=True,
                         metric_for_best_model="accuracy")
trainer = Trainer(model, args, train_dataset=tok_train, eval_dataset=tok_valid,
                  compute_metrics=compute_accuracy,
                  data_collator=DataCollatorWithPadding(bert_tokenizer))
trainer.train()
```

map() runs the tokenizer (inside tokenize_batch) over batches of examples and adds fields such as input_ids and attention_mask; DataCollatorWithPadding pads each batch; compute_metrics receives an object with predictions and label_ids. The Trainer handles batching, shuffling, evaluation, checkpoints, logging and multi-GPU training.

## 11. What is teacher forcing in an encoder-decoder model, and how does the chapter's nmt_collate_fn set it up?

During training, the decoder's input at each step is the correct previous target token, whatever it actually predicted. This significantly speeds up training and improves performance: the decoder can process the whole target sequence in a single call, and an early mistake can't derail the rest of the sequence. The first decoder input is a start-of-sequence (SoS) token, and the model learns to finish with an end-of-sequence (EoS) token, so it knows when to stop.

nmt_collate_fn wraps each Spanish target in SoS and EoS, tokenizes it, then offsets the decoder inputs and the labels by one position:

```python
inputs = NmtPair(src_token_ids, src_mask,
                 tgt_token_ids[:, :-1], tgt_mask[:, :-1])  # drop last position
labels = tgt_token_ids[:, 1:]                             # drop the SoS token
```

So the decoder reads "SoS Me gusta el fútbol" and must output "Me gusta el fútbol EoS". At inference there are no targets, so it gets its own previous output instead; scheduled sampling narrows this gap by gradually switching to the model's own outputs during training.

## 12. Sketch the chapter's GRU encoder-decoder translation model, and explain how it translates a sentence at inference time.

Both languages share one BPE tokenizer and one nn.Embedding (English and Spanish share many words and subwords), and the encoder and decoder are separate 2-layer nn.GRU modules:

```python
def forward(self, pair):
    src_emb = self.embed(pair.src_token_ids)
    tgt_emb = self.embed(pair.tgt_token_ids)
    lengths = pair.src_mask.sum(dim=1).cpu()
    src_packed = pack_padded_sequence(src_emb, lengths=lengths,
                                      batch_first=True, enforce_sorted=False)
    _, hidden_states = self.encoder(src_packed)  # final state of each layer
    outputs, _ = self.decoder(tgt_emb, hidden_states)  # used as initial state
    return self.output(outputs).permute(0, 2, 1)  # [batch, vocab, time]
```

The encoder's final hidden states initialize the decoder, and they're its only information about the source sentence. The decoder inputs aren't packed; instead, nn.CrossEntropyLoss(ignore_index=0) ignores the positions whose target is padding. To translate, decode greedily: feed the source plus just the SoS token, append the most likely next token to the decoder input, and repeat until the model outputs EoS or a maximum length is reached.

## 13. Compare Bahdanau and Luong attention: how does each score an encoder output, and which decoder state does each use?

Both compute, at each decoder step t, a score e₍ₜ,ᵢ₎ for every encoder output ŷ₍ᵢ₎, turn the scores into weights with a softmax, α₍ₜ,ᵢ₎ = exp(e₍ₜ,ᵢ₎) / ∑ᵢ′ exp(e₍ₜ,ᵢ′₎), and return the weighted sum ∑ᵢ α₍ₜ,ᵢ₎ ŷ₍ᵢ₎. They differ in the scoring:

- Bahdanau (concatenative, or additive) attention concatenates the decoder's previous hidden state with each encoder output and scores the pair with a small dense layer: e₍ₜ,ᵢ₎ = vᵀ tanh(W[h₍ₜ₋₁₎; ŷ₍ᵢ₎]). Its output feeds into the decoder's recurrent step.
- Luong (multiplicative) attention uses the decoder's current state h₍ₜ₎ and a dot product, e₍ₜ,ᵢ₎ = h₍ₜ₎ᵀ ŷ₍ᵢ₎ (both vectors must have the same size), or the "general" variant h₍ₜ₎ᵀ W ŷ₍ᵢ₎. The attention output is concatenated with h₍ₜ₎ to predict the next token, so attention stays outside the decoder's recurrence, which is simpler and faster.

Dot products are cheap on modern hardware and performed better in Luong's experiments, so concatenative attention is now rarely used.

## 14. Write the chapter's dot-product attention function in PyTorch with its tensor shapes, and explain how the translation model uses it.

It works like a soft dictionary lookup: compare each query with every key, softmax the similarity scores into weights, and return the weighted sum of the values:

```python
def attention(query, key, value):  # [B, Lq, d], [B, Lk, d], [B, Lk, dv]
    scores = query @ key.transpose(1, 2)     # [B, Lq, Lk]
    weights = torch.softmax(scores, dim=-1)  # each row sums to 1
    return weights @ value                   # [B, Lq, dv]
```

torch.bmm() computes the same batched matrix products for 3D tensors, a bit faster. In the translation model, the queries are the decoder's outputs (equal to its top layer's hidden states), and the keys and values are both the encoder's outputs. Because the encoder read a packed sequence, its outputs are packed too, so they first go through pad_packed_sequence(). The attention output is concatenated with the decoder outputs along the last dimension, so the output layer's input size doubles to 2 × hidden_dim. Two limitations: this version doesn't mask the padding tokens in the source, and it computes Lq × Lk weights, which grows quadratically with sentence length.
