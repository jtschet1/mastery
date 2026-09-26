# Chapter 16 — Vision and Multimodal Transformers

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. The author
hasn't published solutions for this chapter yet (the exercise section of
`handson-mlp-main/16_vision_and_multimodal_transformers.ipynb` says it's a
work in progress), so the questions below have no answers. build_cards.py
skips a card until it has one, so write yours under each heading as you
read.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

### Hands-on exercises (not flashcards; do these in a notebook)

Summarized here; see the book for the full text.

- 12. Fine-tune a pretrained ViT on Food 101 (torchvision.datasets.Food101);
  compare with zero-shot CLIP.
- 13. Build a search engine for your own photos with CLIP embeddings (text
  or image queries; hand-rolled similarity search, FAISS, or a vector
  database).
- 14. Use BLIP-2 to caption all of your photos.

## 1. Can you describe the original ViT’s architecture? Why does it matter?

## 2. What tasks are regular ViTs (meaning nonhierarchical) best used for? What are their limitations?

## 3. What is the main innovation in DeiT? Is this idea generalizable to other architectures?

## 4. What are some examples of hierarchical ViTs? What kind of tasks are they good for?

## 5. How do PVTs and Swin Transformers reduce the computational cost of processing high-resolution images?

## 6. How does DINO work? What changed in DINOv2? When would you want to use DINOv2?

## 7. What is the objective of the JEPA architecture? How does it work?

## 8. What is a multimodal model? Can you give five examples of multimodal tasks?

## 9. Explain what the fusion and alignment problems are in multimodal learning. Why are transformers well suited to tackle them?

## 10. Can you write a one-line summary of the main ideas in VideoBERT, ViLBERT, CLIP, DALL·E, Perceiver IO, Flamingo, and BLIP-2?

## 11. If you are using a Perceiver IO model and you double the length of the inputs and the outputs, approximately how much more computation will be required?
