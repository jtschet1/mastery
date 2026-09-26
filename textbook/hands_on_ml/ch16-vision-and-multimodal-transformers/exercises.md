# Chapter 16 — Vision and Multimodal Transformers

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). Questions are copied verbatim from the book. The author
hasn't published solutions for this chapter yet (the exercise section of
`handson-mlp-main/16_vision_and_multimodal_transformers.ipynb` says it's a
work in progress), so the answers below were written by Claude from the
chapter text, not by the author. If the author publishes solutions later,
swap theirs in.

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

The original vision transformer (ViT) treats an image like a sentence, with patches playing the role of words ("An Image Is Worth 16 × 16 Words"):
1. The image is chopped into 16 × 16 patches: a 224 × 224 RGB image gives a 14 × 14 grid of 196 patches.
2. Each patch is flattened into a 16 × 16 × 3 = 768-dimensional vector and linearly projected to the model's embedding size (a Conv2d layer with kernel size and stride both equal to 16 does exactly this).
3. A trainable class token is prepended to the sequence, and learnable positional embeddings are added.
4. The sequence goes through a regular encoder-only transformer, and a classification head on top of the class token's output makes the prediction, just like BERT-style classification.

It matters because it was the first vision transformer without any CNN, and it showed that a plain transformer could beat the state of the art on ImageNet classification, as long as it was trained on enough data (the authors used 300 million extra images, because transformers lack the inductive biases of CNNs). It kicked off a wave of vision transformers (DeiT, PVT, Swin, DINO, and more), and it showed that images can be tokenized and processed just like text, paving the way for multimodal transformers.

## 2. What tasks are regular ViTs (meaning nonhierarchical) best used for? What are their limitations?

Regular ViTs, such as the original ViT and DeiT, are best used for image classification, and more generally for producing a global representation of an image from the class token's output (e.g., a ViT can serve as CLIP's image encoder). Their main limitations are:
- They output a single-scale, coarse sequence of patch tokens (e.g., one per 16 × 16 patch) rather than multiscale feature maps, so they're not well suited to dense prediction tasks such as object detection or semantic segmentation, which need fine-grained spatial resolution.
- Attention is quadratic in the number of patches: doubling the image's width and height quadruples the number of patches and multiplies the computation by 16. So high-resolution images must be downsampled first, which can hurt accuracy, especially for dense prediction tasks.
- They have fewer inductive biases than CNNs (e.g., locality and translation invariance), so they need a lot of training data (the original ViT used 300 million extra images) unless you use tricks such as distillation (as in DeiT) or self-supervised pretraining, or start from a pretrained model.

## 3. What is the main innovation in DeiT? Is this idea generalizable to other architectures?

DeiT (data-efficient image transformer) keeps the original ViT architecture but trains it using knowledge distillation from a strong, frozen teacher: a state-of-the-art CNN. Its main innovation is the distillation token: a trainable token added to the input sequence alongside the class token, whose output goes through its own classification head. The class token's head is trained with the cross-entropy on the true labels (hard targets), while the distillation head is trained with the cross-entropy on the teacher's predictions (soft targets), and the final loss is a weighted sum of both (typically with equal weights). This allowed DeiT to reach competitive results on ImageNet without any extra training data, whereas the original ViT needed hundreds of millions of extra images.

Yes, the idea is generalizable. Distillation itself works for all kinds of models and tasks (e.g., DistilBERT in NLP), and the teacher doesn't even need to share the student's architecture: in DeiT, a CNN teaches a transformer. The distillation token trick can be applied to any transformer that uses a class token, since it only requires adding one token and one head.

## 4. What are some examples of hierarchical ViTs? What kind of tasks are they good for?

Examples include the Pyramid Vision Transformer (PVT), the Swin Transformer and Swin v2, as well as Twins-SVT, FocalNet, MaxViT and InternImage. Like a CNN, a hierarchical ViT processes the image into a pyramid of gradually smaller but deeper (semantically richer) feature maps: it starts with small patches (e.g., 4 × 4 pixels) to get a high spatial resolution, then works on coarser and coarser tokens with more channels at each level. For example, given a 256 × 192 image, PVT outputs 64 × 48, 32 × 24, 16 × 12 and 8 × 6 feature maps with 64, 128, 320 and 512 channels. To cope with the many small patches, these models use cheaper attention variants, such as PVT's spatial reduction attention or Swin's shifted-window attention.

These multiscale feature maps make hierarchical ViTs great for dense prediction tasks such as object detection, semantic segmentation and instance segmentation: they can replace the CNN backbone of existing architectures, for example in an FCN-style segmentation model or a Mask R-CNN. They also do well at image classification, and Swin's linear scaling makes it well suited to large, high-resolution images.

## 5. How do PVTs and Swin Transformers reduce the computational cost of processing high-resolution images?

Both use small patches to preserve fine spatial resolution, which means many tokens, so regular multi-head attention, which is quadratic in the number of tokens, would be far too expensive. They tackle this differently:
- PVT uses spatial reduction attention (SRA): the queries keep their full resolution, but the keys and values are first spatially downsampled by a factor R in each dimension (usually with a strided convolutional layer followed by layer norm), which divides the number of attention scores by R². At PVT's first level, the 3,072 tokens of a 64 × 48 grid attend to keys and values reduced 8 times horizontally and vertically, to an 8 × 6 grid of 48 tokens: that's 3,072 × 48 = 147,456 scores instead of over 9 million, 64 times fewer, with no loss of output resolution. The cost is still quadratic in the image area, though.
- Swin uses window-based multi-head self-attention (W-MSA): each patch only attends to the patches within the same small, non-overlapping window (e.g., 7 × 7 = 49 patches), so the cost grows linearly with the image area: doubling the width and height multiplies it by 4 instead of 16. To let information flow between windows, every other layer uses shifted windows (SW-MSA), offset by half a window, which is implemented efficiently by cyclically shifting the image and using attention masks.

In both cases, the pyramid structure also reduces the number of tokens at each deeper level.

## 6. How does DINO work? What changed in DINOv2? When would you want to use DINOv2?

DINO (self-distillation with no labels) is a self-supervised technique for learning image representations. During training, the model is duplicated into a student and a teacher: only the student is trained by gradient descent, while the teacher's weights are an exponential moving average of the student's (a momentum teacher). Each image is augmented differently for each of them (color jitter, grayscale, Gaussian blur, flips, and so on): the teacher sees the full image with mild augmentations, while the student often sees just a zoomed-in part with stronger augmentations. The student is trained to match the teacher's predictions, which forces both to agree on high-level representations. To avoid mode collapse, where both output the same thing regardless of the input, DINO centers the teacher's logits by subtracting their moving average, and sharpens them using a low temperature. After training, you keep a single network: the student, or the momentum teacher, which the DINO authors found performs even better. Its class token output is an excellent image representation (e.g., for nearest-class-mean classification), and its attention maps often segment the main object, without ever seeing a label.

DINOv2 was trained on a much larger, curated dataset, and tweaked to output per-patch features, not just a global representation. So you'd want to use DINOv2 as a general-purpose, pretrained vision foundation model, e.g., as a feature extractor or backbone for classification and for dense prediction tasks such as segmentation, especially when you have few labels.

## 7. What is the objective of the JEPA architecture? How does it work?

The joint-embedding predictive architecture (JEPA), proposed by Yann LeCun as part of his world-model framework, aims to learn meaningful representations that deepen an AI's understanding of the world and make its predictions more reliable. Its objective is to predict the missing parts of an input in embedding space, rather than in pixel space.

During training, JEPA uses two encoders and a predictor. The teacher encoder sees the full input (e.g., a photo of a cat), while the student encoder only sees part of it (e.g., the same photo without the cat's ears). Both encoders produce embeddings, and the predictor must predict the teacher's embeddings for the missing part (the ears), given the student's embeddings for the visible part. The student encoder and the predictor are trained jointly, while the teacher encoder is just a moving average of the student encoder, much like in DINO. Since it predicts abstract embeddings rather than every pixel, JEPA is fast, parameter-efficient and learns more semantic features. After training, the teacher encoder and the predictor are dropped, and the student encoder is used to produce representations for downstream tasks. I-JEPA implements this for images, while V-JEPA and V-JEPA 2 process videos.

## 8. What is a multimodal model? Can you give five examples of multimodal tasks?

A multimodal model is a model that can handle multiple modalities, meaning different kinds of data such as text, images, audio, video, or robot sensor and actuator signals, and capture how they interact. This is challenging because modalities are heterogeneous (continuous or discrete, temporal or spatial, high or low resolution, noisy or clean), and they can carry overlapping information (e.g., lip movements and speech) or combine into a new meaning (e.g., words said while rolling one's eyes). Examples of multimodal tasks:
- Image or video captioning.
- Visual question answering: answering a text question about an image.
- Image search from a text query (or from an image query), e.g., with CLIP embeddings.
- Text-to-image generation, e.g., with DALL·E or Stable Diffusion.
- Speech-to-text and text-to-speech.
- Visual grounding: locating the object described by a text query, such as "the dog next to the tree".
- Embodied AI: a model that physically interacts with its environment, e.g., a robot following instructions.

## 9. Explain what the fusion and alignment problems are in multimodal learning. Why are transformers well suited to tackle them?

- Fusion is the problem of combining different modalities so the model can use them jointly, for example by encoding them into the same representation space. It's hard because modalities are very heterogeneous (e.g., discrete text tokens versus continuous, high-resolution audio or pixels).
- Alignment is the problem of discovering the relationships between modalities: for example, finding the timestamp of each word of a transcript in a speech recording, or finding the most relevant object in an image given a text query such as "the dog next to the tree" (visual grounding).

Transformers are well suited to both. First, they can ingest pretty much any modality, as long as you can chop it into a sequence of meaningful tokens (e.g., words, image patches, audio or video clips) and embed them. Embeddings from different modalities can then be fused in various ways: summed, concatenated into a single sequence, passed through a fusion encoder, or processed by separate encoders that exchange information through cross-attention (e.g., co-attention). Second, multi-head attention is a powerful tool to detect and exploit complex patterns, both within and across modalities, which takes care of alignment: through cross-attention, tokens of one modality can attend to the relevant tokens of another.

## 10. Can you write a one-line summary of the main ideas in VideoBERT, ViLBERT, CLIP, DALL·E, Perceiver IO, Flamingo, and BLIP-2?

- VideoBERT: a pretrained BERT extended to video by turning short clips into discrete visual tokens (3D CNN features clustered with hierarchical k-means), trained with masked token prediction on text or video, and with text-video alignment on both concatenated in a single stream.
- ViLBERT: a dual-stream text-plus-image model: a BERT-based text encoder and a visual encoder over Faster R-CNN region features exchange information through co-attention (two-way cross-attention) layers, pretrained with masked prediction and image-text alignment.
- CLIP: image and text encoders trained with a contrastive loss on 400 million image-caption pairs so that matching pairs get similar embeddings, enabling zero-shot image classification and image search.
- DALL·E: a GPT-like model trained with next token prediction on text tokens followed by discrete image tokens (from a dVAE), so it can generate an image token by token from a text prompt.
- Perceiver IO: a modality-agnostic model in which a short array of learned latent tokens reads arbitrarily long raw inputs via cross-attention, and output query tokens read the latents the same way, so it scales linearly with input and output sizes and handles many tasks.
- Flamingo: connects a frozen vision encoder and a frozen LLM through a Perceiver Resampler and tanh-gated cross-attention layers, enabling open-ended visual dialogue with excellent few-shot performance on interleaved images and text.
- BLIP-2: bridges a frozen image encoder and a frozen LLM with a lightweight querying transformer (Q-Former), trained first with image-text matching, contrastive and captioning objectives, then to map its query outputs into the LLM's input space.

## 11. If you are using a Perceiver IO model and you double the length of the inputs and the outputs, approximately how much more computation will be required?

Roughly twice as much, at most. In Perceiver IO, the inputs are only accessed through cross-attention from a fixed-size array of N latent tokens, so reading M input tokens costs about M × N attention scores: that's linear in the input length. The latent transformer blocks only process the N latent tokens, so their cost doesn't depend on the input or output lengths at all. And the O output query tokens only attend to the latent tokens, again through cross-attention, which costs about O × N: linear in the output length. So doubling both the input and output lengths doubles the cost of the input and output cross-attention layers (and of the per-token projections), while the latent processing cost stays the same: overall, the computation is multiplied by about 2, or a bit less. By comparison, if a regular transformer processed the same sequences with self-attention, doubling their length would multiply the cost of its attention layers by about 4, since attention is quadratic in the sequence length.
