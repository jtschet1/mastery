# Chapter 16 — Vision and Multimodal Transformers

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's end-of-chapter questions are in
exercises.md, with answers written by Claude, since the author hasn't
published any; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. What is an inductive bias? Give examples from common architectures, and explain how inductive biases trade off against the amount of training data.

An inductive bias is an assumption about the data that a model makes implicitly because of its architecture:
- Linear models assume the data is linear.
- CNNs assume locality (nearby pixels are strongly related) and that a pattern learned in one location is useful everywhere else.
- RNNs assume the inputs are ordered and that recent inputs matter more than older ones.
- Even chopping an image into patches injects a bias toward proximity: nearby pixels are assumed to be more strongly correlated than distant ones.

Correct biases act as built-in knowledge, so the model needs less training data; wrong ones hurt performance however much data you have. Transformers make few assumptions, so they must learn such regularities from the data and need far more of it than CNNs. Given enough data, though, a low-bias model is flexible enough to discover patterns that a strongly biased one would miss.

## 2. How do you turn a batch of images into a ViT's input tokens in PyTorch (patch embeddings, class token, positional embeddings)?

A Conv2d whose kernel_size and stride both equal the patch size is equivalent to chopping the image into non-overlapping patches, flattening each one, and applying the same linear layer to all of them. Its output is a feature map, so flatten the spatial dimensions and move the embedding dimension last:

```python
patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
Z = patch_embed(images)  # [B, E, 14, 14] for 224 × 224 images
Z = Z.flatten(start_dim=2).transpose(1, 2)  # [B, 196, E]
cls = cls_token.expand(Z.shape[0], -1, -1)  # [1, 1, E] → [B, 1, E]
Z = torch.cat((cls, Z), dim=1)  # [B, 197, E]
Z = Z + pos_embed  # pos_embed is [1, 197, E], broadcast across the batch
```

cls_token and pos_embed are nn.Parameter tensors initialized with small random values (e.g., a standard deviation of 0.02). The tokens then go through an nn.TransformerEncoder (built with batch_first=True, here with activation="gelu"), and only the class token's output, Z[:, 0], is normalized and passed to the classification head.

## 3. Walk through fine-tuning a pretrained ViT for image classification with the Hugging Face Trainer, including the gotchas.

1. Load the model with a new head: ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=37) gives the model a new, untrained classification head for your number of classes.
2. Load the matching preprocessor with AutoImageProcessor.from_pretrained(model_id, use_fast=True). It resizes each image to 224 × 224, scales pixel values to between −1 and 1, moves the channels first, and returns a dict with a "pixel_values" entry.
3. Write a collate function that runs the processor on a batch of images (return_tensors="pt", plus do_convert_rgb=True because some images are RGBA and would crash training) and adds a "labels" tensor.
4. Create TrainingArguments (output directory, batch size, number of epochs, eval_strategy="epoch", remove_unused_columns=False), pass them to a Trainer with the model, collate function and datasets, and call trainer.train().

Why remove_unused_columns=False: by default, the Trainer drops the dataset columns that the model's forward() method doesn't accept, such as "image", before your collate function gets to see them.

## 4. Compare BEiT and MAE, two ways to pretrain ViTs with masked image modeling.

Both borrow BERT's idea: hide some image patches and train the model to reconstruct them from the visible ones, with no labels needed.
- BEiT doesn't predict pixels. A discrete variational autoencoder (dVAE) first turns each patch into a visual token ID from a fixed vocabulary, and BEiT must predict the IDs of the masked patches, just like masked language modeling. This avoids wasting capacity on unimportant pixel details, but it requires a separately trained tokenizer.
- MAE (masked autoencoder) removes the dVAE and predicts raw pixel values directly. Its encoder-decoder is asymmetric: a large encoder processes only the visible patches, and a lightweight decoder reconstructs the whole image. Since about 75% of the patches are masked, the encoder handles only a quarter of the tokens, which makes pretraining much cheaper and lets it scale to very large datasets. Afterward, only the encoder is kept for downstream tasks.

## 5. What's the difference between single-stream and dual-stream multimodal transformers, and how does ViLBERT's co-attention connect its two streams?

A single-stream model, such as VideoBERT, fuses the modalities early: it concatenates the text tokens and visual tokens into one sequence processed by a single encoder. It's simple, but it treats both modalities identically.

A dual-stream model, such as ViLBERT, gives each modality its own encoder, so each gets the processing it needs, and a model pretrained on text alone isn't forced to digest foreign inputs that could damage its weights. In ViLBERT, the text stream starts with BERT layers, while the visual features come from a frozen, pretrained Faster R-CNN (one vector per detected region) and are already high level, so the visual stream only needs co-attention layers.

The streams are connected by pairs of co-attention layers: in each pair, one stream provides the queries and the other the keys and values, and vice versa. This two-way cross-attention lets each modality refine its representations using the other.

## 6. Walk through CLIP's contrastive training loss for a batch of m image-caption pairs.

1. Encode the m images and m captions, project both into a shared space with the same dimensionality, and ℓ2-normalize every vector.
2. Compute the m × m matrix of cosine similarities, where entry (i, j) compares image i with caption j.
3. Divide the similarities by a learned temperature to get logits.
4. Treat each row as an m-way classification whose correct class is the diagonal entry (caption i goes with image i), and compute its cross-entropy; do the same for each column, and average.

This pulls matching pairs together and pushes mismatched pairs apart. Ideally, matches score near +1 and mismatches near 0 rather than −1, since unrelated high-dimensional vectors are nearly orthogonal. Every other caption in the batch serves as a negative example, so the method needs very large batches (CLIP used 32,768 pairs); with too few negatives, the model can overfit details of the positive pairs.

## 7. How do you use CLIP for zero-shot image classification with the Transformers library, both via a pipeline and by hand?

The pipeline does everything in one call:

```python
from transformers import pipeline

clip = pipeline(task="zero-shot-image-classification",
                model="openai/clip-vit-base-patch32")
results = clip(image_url, candidate_labels=["cricket", "ladybug", "spider"],
               hypothesis_template="This is a photo of a {}.")
```

By hand, call a CLIPProcessor on the captions and the image (text=captions, images=[image], return_tensors="pt", padding=True), pass the result to a CLIPModel, and read outputs.image_embeds and outputs.text_embeds, which are already ℓ2-normalized. Then image_embeds @ text_embeds.T gives the cosine similarities; multiply them by clip_model.logit_scale.exp() (the learned inverse temperature, about 100) and apply a softmax to get probabilities. If you encode images and text separately with get_image_features() and get_text_features(), normalize the features yourself.

Caption-like prompts such as "This is a photo of a ladybug" work better than bare labels because CLIP was trained on web captions; if you don't pass a hypothesis_template, the pipeline still wraps each label in a similar caption template.

## 8. How does Flamingo add visual inputs to a frozen pretrained LLM without disrupting it at the start of training?

It inserts gated xattn-dense modules between the frozen LLM's blocks. Each contains a cross-attention layer, whose queries come from the text tokens and whose keys and values are visual tokens (produced by a Perceiver Resampler from the frozen vision encoder's outputs), followed by a feedforward module, each with a skip connection.

The trick is tanh gating: the outputs of the cross-attention layer and of the feedforward module are multiplied by tanh(α), where α is a learnable scalar (one per gate) initialized to 0. Since tanh(0) = 0, at first everything flows through the skip connections and the model behaves exactly like the original LLM. During training, the gates gradually open and visual information starts to influence the text.

The cross-attention is also masked so that each text token only sees the visual tokens of the closest preceding image; earlier images remain reachable indirectly through the LLM's own self-attention.

## 9. BLIP-2 bridges a frozen image encoder and a frozen LLM with a Q-Former. How is it trained?

Stage 1: the Q-Former (initialized from BERT-base, plus new cross-attention layers that attend to the frozen image encoder's visual tokens) processes a caption and a set of learnable query tokens. It's trained on three objectives, each with its own attention mask:
- Image-text matching: queries and text attend to each other, and a binary head predicts whether the caption matches the image (using hard negatives).
- Image-text contrastive: queries and text can't see each other, and a CLIP-like loss aligns the query outputs with the text's class token output.
- Captioning: text tokens attend causally to earlier text and to all the queries (which can't see the text), and the model is trained with next-token prediction.

Stage 2: a new linear layer projects the query outputs into the frozen LLM's input embedding space, where they're placed before the text tokens as a visual prompt, and the model learns to predict the caption's next tokens.
