# Chapter 15 — Transformers for Natural Language Processing and Chatbots

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. The author
hasn't published solutions for this chapter yet (the exercise section of
`handson-mlp-main/15_transformers_for_nlp_and_chatbots.ipynb` says it's a
work in progress), so the questions below have no answers. build_cards.py
skips a card until it has one, so write yours under each heading as you
read.

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

## 2. Why does the Transformer architecture need positional encodings?

## 3. What tasks are encoder-only models best at? How about decoder-only models? And encoder-decoder models?

## 4. What is the most important technique used to pretrain BERT?

## 5. Can you name four BERT variants and explain their main benefits?

## 6. What is the main task used to pretrain GPT and its successors?

## 7. The generate() method has many arguments, including do_sample, top_k, top_p, temperature, and num_beams. What do these five arguments do?

## 8. What is prompt engineering? Can you describe five prompt engineering techniques?

## 9. What are the main steps to build a chatbot, starting from a pretrained decoder-only model?

## 10. How can a chatbot use tools like a calculator or web search?

## 11. What is MCP used for?
