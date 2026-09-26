# Chapter 15 — Transformers for Natural Language Processing and Chatbots

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. The author
hasn't published solutions for this chapter yet (the exercise section of
`handson-mlp-main/15_transformers_for_nlp_and_chatbots.ipynb` says it's a
work in progress), so the answers below were written by Claude from the
chapter text, not by the author. If the author publishes solutions later,
swap theirs in.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text.

- 12. Fine-tune BERT for sentiment analysis on the IMDb dataset.
- 13. Fine-tune GPT-2 on the Shakespeare dataset (from Chapter 14), then
  generate Shakespeare-like text.
- 14. Embed the Wikipedia Movie Plots dataset with SBERT and write a
  function that finds the movies most similar to a search query.
- 15. Build a movie-expert chatbot on an instruction-tuned model (e.g.,
  Qwen-7B-Instruct), then add RAG by injecting the most relevant movie plot
  into the prompt.

## 1. What is the most important layer in the Transformer architecture? What is its purpose?

The most important layer is the multi-head attention (MHA) layer. Every other layer (embeddings, dense layers, layer norm) processes each token independently, so MHA is where tokens exchange information: its purpose is to update each token's representation based on the tokens it attends to, turning vague token embeddings into contextualized ones. For example, in "I like soccer", attending to "I" helps the model infer that "like" is a verb meaning "to be fond of". The encoder uses MHA for self-attention over the whole input sentence, the decoder uses a masked (causal) version in which each token only attends to itself and earlier tokens, and the decoder's cross-attention layers let each target token attend to the encoder's outputs (e.g., "el" attending to "soccer" before "fútbol" is predicted). Each head computes scaled dot-product attention, softmax(QKᵀ / √d_k)·V, on its own linear projections of the queries, keys and values, so different heads can focus on different characteristics of the tokens (such as tense or meaning); their outputs are concatenated and linearly projected. Since attention connects all positions directly and in parallel, transformers capture long-range patterns much better than RNNs and are easy to parallelize.

## 2. Why does the Transformer architecture need positional encodings?

Because all of the Transformer's layers are position-agnostic: attention layers and dense layers treat all positions the same way (unlike recurrent or convolutional layers), so when a layer processes a token, it has no idea where that token is located in the sentence or relative to the other tokens. Without positional information, the model would see each sentence as a bag of tokens (in an encoder, shuffling the input tokens would just shuffle the outputs), yet word order matters for meaning. So we give the model positional encodings: dense vectors representing each position, added to the token embeddings (the ith positional encoding is added to the embedding of the ith token of every sentence). They can be trainable, e.g., an nn.Embedding layer or an nn.Parameter matrix (as in BERT), or fixed, like the sine/cosine scheme proposed in the original paper, which doesn't really perform better than trainable ones. Newer approaches such as relative position bias (RPB), rotary positional encoding (RoPE) and attention with linear bias (ALiBi) generally perform better.

## 3. What tasks are encoder-only models best at? How about decoder-only models? And encoder-decoder models?

- Encoder-only models (e.g., BERT) are best at natural language understanding tasks: text classification (e.g., sentiment analysis), token classification (e.g., named entity recognition), sentence-pair tasks (e.g., natural language inference or paraphrase detection), multiple-choice and extractive question answering, and text embeddings for semantic search, clustering or similarity. They read the whole input bidirectionally in a single pass, so they're fast and often match much larger models on these tasks. But they're not used for text generation, since every new token would require recomputing everything.
- Decoder-only models (e.g., GPT) are best at text generation: auto-completion, creative writing, code generation, free-form question answering, some math and logical reasoning, and chatbots. Because they're causal, they can cache their previous state and generate efficiently one token at a time, and large ones can also handle classification, translation or summarization through zero-shot or few-shot prompting.
- Encoder-decoder models (e.g., the original Transformer, T5, BART) are best at turning an input text into a new output text, especially translation and summarization: the bidirectional encoder builds excellent contextual embeddings of the source text for the decoder, typically giving better results than a decoder-only model of similar size. This architecture is also common for vision tasks with multiple outputs (e.g., object detection) and for multimodal models.

## 4. What is the most important technique used to pretrain BERT?

Masked language modeling (MLM), a self-supervised "fill in the blanks" (cloze) task: each input token has a 15% probability of being selected, and the model must predict the original selected tokens from their context, with the loss computed only on those positions. Most selected tokens are replaced with a [MASK] token, but 10% are replaced with a random token and 10% are left unchanged. The random tokens force the model to perform well even when there are no mask tokens, which is the case in most downstream tasks, and the unchanged tokens encourage it to pay attention to the actual token at the predicted position instead of learning to ignore it. Since the encoder sees the whole sentence at once, MLM lets BERT learn deep bidirectional representations from a large unlabeled corpus. BERT was also pretrained with next sentence prediction (NSP), but it turned out not to help much, so most later models dropped it. RoBERTa also showed the benefit of dynamic masking, where each text is masked differently at each epoch rather than once before training.

## 5. Can you name four BERT variants and explain their main benefits?

Here are five popular ones:
- RoBERTa (Facebook AI): better performance than BERT across the board, mostly thanks to pretraining longer on more data, using dynamic masking (tokens are masked on the fly, differently at each epoch) and dropping NSP.
- DistilBERT (Hugging Face): about 40% smaller and 60% faster than BERT while retaining about 97% of its performance, great for low-resource devices, low latency or quick fine-tuning. It was distilled from BERT: trained on the teacher's temperature-softened predictions, plus the MLM loss and a loss aligning its final hidden states with the teacher's.
- ALBERT (Google): shares the same weights across all encoder layers and factorizes the embedding matrix (small embeddings projected up by a linear layer), making it much smaller than BERT (though not faster), which is handy when memory is limited. It also replaced NSP with sentence order prediction, which gave better sentence embeddings.
- ELECTRA (Google): pretrained with replaced token detection: a small generator fills in masked tokens and the main model (the discriminator) must spot which tokens were replaced. Since it learns from every token, not just the masked ones, it's more sample-efficient and converges faster, matching larger BERT models.
- DeBERTa (Microsoft): uses relative positional embeddings inside every attention layer (disentangled attention) instead of absolute positional embeddings, beating the state of the art on many NLU tasks; DeBERTaV3 adds ELECTRA-style pretraining.

## 6. What is the main task used to pretrain GPT and its successors?

Next token prediction (NTP), also called causal language modeling: the model is fed sequences of text from a huge unlabeled corpus and trained to predict the next token at every position, with a causal mask so each position can only attend to the tokens before it. For example, given "Happy birthday", it should predict "birthday to": "birthday" after "Happy", and "to" after "birthday". This is self-supervised, since the targets are just the inputs shifted by one token, and it's simple and efficient: GPT-1 was trained on 512-token sequences sampled from a corpus of books, with no padding or special tokens at all, and every position provides a training signal. A model pretrained this way can generate text one token at a time, appending each predicted token to its input, which is why GPT-2, GPT-3 and today's base LLMs are typically pretrained with NTP before being fine-tuned for chat and instruction following.

## 7. The generate() method has many arguments, including do_sample, top_k, top_p, temperature, and num_beams. What do these five arguments do?

- do_sample: by default (False), generate() uses greedy decoding, always picking the most likely next token. That's fine for structured outputs or question answering, but for creative writing it often makes the model repeat itself or get stuck in a loop. With do_sample=True, each token is sampled randomly according to the model's estimated probabilities instead.
- temperature (default 1): the logits are divided by the temperature before the softmax. A lower temperature makes the output more predictable (close to 0, sampling approaches greedy decoding), while a higher one makes it more diverse, and eventually incoherent.
- top_k: only sample from the k most likely next tokens.
- top_p (nucleus sampling): only sample from the smallest set of most likely tokens whose total probability is at least top_p. It's often preferred over top-k because it adapts to the distribution: after "The capital city of France is", the set contains essentially just "Paris", while after "My favorite city is", it includes many plausible cities, whereas top-k would sometimes allow bad tokens in the first case and exclude good ones in the second.
- num_beams: the beam width for beam search (default 1, meaning no beam search). Beam search keeps the num_beams most likely partial sequences, extends each by one token at every step, and keeps the best ones, so an early mistake can still be corrected later.

Note that temperature, top_k and top_p only have an effect when sampling (do_sample=True).

## 8. What is prompt engineering? Can you describe five prompt engineering techniques?

Prompt engineering is the art of crafting and tweaking a prompt until the model reliably behaves the way you want. LLMs are very sensitive to phrasing, so it's worth experimenting, even programmatically by evaluating many prompt variants. Five techniques:
1. Clear, well-framed instructions: choose your words carefully, add context, give the model a persona to imitate (e.g., "You are a friendly real-estate expert"), specify the output format and style, and list pitfalls to avoid.
2. Few-shot prompting (in-context learning): include a few examples of the task in the prompt, such as "Capital city of France = Paris" before asking about another country, so the model infers the task and the expected format.
3. Prompt chaining: break a complex task into subtasks, with one prompt each, feeding each output into the next prompt (e.g., write a lesson outline, then check and complete it, then write the lesson from it).
4. Chain-of-thought prompting: ask the model to reason step by step, or show it step-by-step example answers. For more reliability, run it several times and keep the most frequent answer (self-consistency); tree-of-thoughts goes further by exploring and evaluating several reasoning branches, with backtracking, at a high cost.
5. Retrieval augmented generation (RAG): retrieve relevant, reliable information (e.g., from a database or search engine) and inject it into the prompt, which greatly reduces hallucinations.

Other options include automatic prompt optimization (e.g., prompt tuning), multi-agent debate, self-critique and refinement, and having an LLM write prompts for another model.

## 9. What are the main steps to build a chatbot, starting from a pretrained decoder-only model?

1. Supervised fine-tuning (SFT): fine-tune the base model on a curated dataset of instructions and responses, conversations, question/answer pairs, code and math problems with solutions, role-play, and safety-aligned answers (e.g., declining to explain how to rob a bank). This is regular next token prediction, usually with loss masking, meaning the loss is only computed on the answer tokens. Multi-turn conversations with role tags (e.g., "User:" and "Assistant:", or the ChatML format) teach the model to hold a conversation. The result is a conversational, instruction-following model.
2. Fine-tuning with human feedback: human raters compare or rank the model's answers, and the model is fine-tuned to produce the preferred ones. RLHF trains a reward model on these preferences, then optimizes the LLM with an RL algorithm (PPO) while keeping it close to the original model; DPO is a simpler, more stable alternative that trains directly on (prompt, chosen answer, rejected answer) triplets against a frozen reference model. Libraries such as TRL implement SFT, RLHF and DPO.
3. Deployment in a full chatbot system: a user interface (web or app) and possibly an API endpoint, storage of the conversations, and an orchestrator that coordinates tools such as a calculator, web search, retrieval augmented generation or long-term memory (e.g., via MCP servers).

Alternatively, you can skip the fine-tuning steps by downloading a model that's already fine-tuned for chat (e.g., Mistral-7B-Instruct), or use a conversational model through an API.

## 10. How can a chatbot use tools like a calculator or web search?

Through an orchestrator, the component of the chatbot system that sits between the user and the model and coordinates the tools. There are two main approaches:
- The orchestrator decides: for example, it detects a math expression in the user's prompt, evaluates it with a calculator, and adds the result to the prompt (e.g., "System: Calculator result = 42") before calling the model, which then only has to phrase the answer. If the user mentions a URL, it can fetch the web page and inject its text into the prompt, or a summary of it, or just its most relevant chunks, found by comparing the chunks' embeddings with the prompt's embedding.
- The model decides (tool augmentation, or function calling): the model is fine-tuned to emit a special tool invocation when needed, such as "[calculator_tool] 525.6 × 315 / 3942 [/calculator_tool]", "[search_tool] What is the population of Ottawa? [/search_tool]", or a JSON message. The orchestrator detects it and runs the tool (e.g., a web search whose top results are fetched and summarized). It can then substitute the result directly into the model's output, or feed it back to the model to write the final answer: this costs an extra call, but the model gets to see the result, so it can comment on it and cite its sources.

The same pattern generalizes to retrieval augmented generation over private data, long-term memory, code interpreters and many other tools, and MCP standardizes how the orchestrator connects to such tool servers.

## 11. What is MCP used for?

The Model Context Protocol (MCP) is an open standard, proposed by Anthropic, for connecting AI systems to external tools and resources, such as file systems, email, calendars, weather or navigation services, each exposed by an MCP server. It doesn't specify anything about the LLM itself: the LLM orchestrator (the MCP host) detects when the LLM wants to use a tool (e.g., because it outputs a JSON request as instructed in its system prompt), sends an MCP-compliant JSON request to the appropriate MCP server through an MCP client, then feeds the server's response back to the LLM so it can compose its answer (e.g., "It will be sunny today in Paris"). Compared with a plain REST or gRPC API, MCP connections are long-lived, stateful and bidirectional, and MCP includes an AI-friendly discovery mechanism: the client can ask a server for a rich textual description of what it does and exactly how to use its functions and parameters (a self-documenting API for AIs), and the server can ask about the client's capabilities, such as displaying images or streaming output. As a result, connecting an LLM to a new service mostly boils down to adding the server to the orchestrator's configuration and telling the LLM about it.
