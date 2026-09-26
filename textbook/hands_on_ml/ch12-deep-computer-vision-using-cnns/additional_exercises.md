# Chapter 12 — Deep Computer Vision Using Convolutional Neural Networks

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's own end-of-chapter exercises are in
exercises.md; the two files together make 20 cards for this chapter.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. What is a neuron's receptive field in a CNN, and why do architectures prefer stacks of 3×3 convolutions over one larger kernel?

A convolutional neuron is connected only to a small patch of the previous layer, its receptive field: with f_h × f_w kernels and strides s_h and s_w, the neuron at row i, column j sees rows i·s_h to i·s_h + f_h − 1 and columns j·s_w to j·s_w + f_w − 1, across all the input feature maps. Stacking layers widens the region of the image each neuron depends on, so low-level features get assembled into larger, more complex patterns higher up.

Two stacked 3×3 layers (stride 1) see the same 5×5 input patch as one 5×5 layer, with fewer parameters (18C² weights instead of 25C² for C channels throughout), fewer computations, and an extra nonlinearity; they usually perform better. The exception is the first layer, where a large kernel (e.g., 7×7) with stride 2 shrinks the image cheaply, since the input has only about 3 channels.

## 2. Describe nn.Conv2d's main arguments, the input and weight shapes it uses, and how to compute its output height and width.

nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0), where out_channels is the number of filters (output feature maps), expects inputs shaped [batch, channels, height, width]; images stored channels-last need .permute(0, 3, 1, 2) first.

- The weight has shape [out_channels, in_channels, kernel_height, kernel_width] and the bias [out_channels]. No image size appears (the kernels are shared across positions), so any image at least as large as the kernel works.
- Each output dimension is ⌊(n + 2p − k) / s⌋ + 1 for input size n, padding p, kernel size k, and stride s (with no dilation).
- The default padding=0 (also written "valid") shrinks the maps: a 7×7 kernel turns 70×120 into 64×114. padding="same" keeps the size but requires stride 1. A 7×7 kernel with stride=2 and padding=3 turns 70×120 into 35×60.
- To size the first nn.Linear after flattening, multiply the last maps' channels × height × width, or use nn.LazyLinear to infer it.

## 3. What do nn.MaxPool2d, nn.AvgPool2d, and nn.AdaptiveAvgPool2d do, and why do modern CNNs use global average pooling before the output layer?

Pooling layers have no parameters: they aggregate small windows of each channel independently, so the number of channels is unchanged.

- nn.MaxPool2d(kernel_size=2) keeps the max of each 2×2 window. The stride defaults to the kernel size and padding to 0, so it halves the height and width (rounding down), saving computation and memory and adding some invariance to small shifts.
- nn.AvgPool2d takes the mean instead. Max pooling usually works better: it keeps the strongest activations, gives more translation invariance, and is slightly cheaper.
- nn.AdaptiveAvgPool2d(output_size=1) picks the kernel size needed to produce the requested output size; with 1, it's global average pooling, the mean of each whole feature map (like X.mean(dim=(2, 3), keepdim=True)).

Global average pooling reduces each feature map to a single number, so the head needs no big dense layers over flattened maps: far fewer parameters, less overfitting, and a head that works whatever the input image size.

## 4. What are 1×1 convolutional layers good for, given that they can't detect spatial patterns?

A 1×1 convolution applies the same small dense layer to every pixel's vector of channel values. It's useful in several ways:

- It captures patterns across channels, along the depth dimension.
- With fewer output than input channels, it acts as a bottleneck layer that reduces dimensionality before an expensive 3×3 or 5×5 convolution, cutting parameters and computation. GoogLeNet's inception modules do this, as do ResNet's deeper residual units (1×1 down to 64 maps, 3×3 with 64 maps, then 1×1 back up to 256).
- Followed by a 3×3 or 5×5 layer, it forms something like a single, more powerful convolutional layer: a two-layer network swept across the image instead of a single linear filter.
- It changes the depth cheaply: with stride 2, it reshapes a ResNet skip connection to match the main path, and in a depthwise separable convolution it's the pointwise step that mixes channels.

## 5. Why do skip connections (residual learning) make very deep networks such as ResNet trainable?

A residual unit adds its input x to the output of a small stack of layers, so to model a target function h(x), those layers only need to learn the residual f(x) = h(x) − x.

- At initialization the weights are small, so a plain stack of layers outputs values close to 0, whereas a residual unit outputs roughly a copy of its input. The network starts close to the identity function, which is often not far from the target, so training is much faster.
- The signal can cross the whole network through the skip connections, so the network makes progress even while some layers haven't started learning; in backprop, the additions also pass gradients straight down to the lower layers.

## 6. Sketch a ResNet-34 residual unit as a PyTorch nn.Module, and say how ResNet-34 stacks these units.

The main path is conv, BN, ReLU, conv, BN. The skip path is the identity, unless the unit downsamples, in which case a strided 1×1 convolution plus BN reshapes the input. forward() adds the two paths, then applies ReLU:

```python
class ResidualUnit(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out))
        self.skip = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))
```

The convolutions have no bias because BN follows them. ResNet-34 starts with a 7×7 stride-2 convolution (with BN and ReLU) and a 3×3 stride-2 max pool, then stacks 3 units with 64 feature maps, 4 with 128, 6 with 256, and 3 with 512, using stride 2 in the first unit of each new size. It ends with global average pooling, flattening, and a dense output layer.

## 7. How does a depthwise separable convolution work, why is it cheaper than a regular one, and how do you build it in PyTorch?

A regular convolutional filter looks for spatial and cross-channel patterns jointly. A depthwise separable convolution assumes they can be modeled separately: a depthwise convolution applies one spatial filter per input channel, then a pointwise (1×1) convolution mixes the channels. With k × k kernels, that's k²·C_in + C_in·C_out weights instead of k²·C_in·C_out: for 3×3 kernels with 256 channels in and out, about 68K instead of 590K. It uses less memory and computation and often performs better (Xception, MobileNet).

PyTorch has no separable layer, but the groups argument splits the input channels into independent groups, each with its own filters; groups=in_channels, with as many output channels as input channels, gives the depthwise part:

```python
separable = nn.Sequential(
    nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in),  # depthwise
    nn.Conv2d(c_in, c_out, kernel_size=1),                         # pointwise
)
```

Avoid it right after layers with few channels, such as the RGB input.

## 8. How do you classify images with a pretrained TorchVision model, from loading the weights to reading off class names?

Pick a weights enum, build the model with it, and preprocess with the transforms that come with those weights:

```python
weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
model = torchvision.models.convnext_base(weights=weights).to(device)
preprocess = weights.transforms()
model.eval()
with torch.no_grad():
    logits = model(preprocess(images).to(device))
top3_logits, top3_ids = logits.topk(k=3, dim=1)
class_names = weights.meta["categories"]
print([class_names[i] for i in top3_ids[0]])  # top 3 for the first image
```

The weights are downloaded once, then cached. weights.transforms() is safer than hand-rolled preprocessing: it resizes and crops images to the size the model expects and standardizes each color channel with the means and standard deviations used in training (ImageNet's). Switch to evaluation mode first, since models start in training mode, and turn off autograd. torchvision.models.list_models() lists the available architectures.

## 9. Walk through adapting a pretrained TorchVision classifier to a new set of classes and fine-tuning it.

1. Preprocess the new images with the transforms that come with the pretrained weights enum (e.g., weights = ConvNeXt_Base_Weights.IMAGENET1K_V1), for instance by passing transform=weights.transforms() to the dataset.
2. Find the head, e.g. with model.named_children() (a ConvNeXt has features, avgpool, and classifier), and replace its output layer with one sized for your classes. For ConvNeXt-Base: model.classifier[2] = nn.Linear(1024, n_classes).to(device).
3. Freeze everything except the head, so the new layer's large early errors don't wreck the pretrained weights:

```python
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
```

4. Train for a few epochs: the new head alone often reaches good accuracy.
5. Unfreeze the pretrained layers (all at once, or gradually from the top), lower the learning rate by about 10×, and keep training. Parameter groups can give lower layers smaller learning rates than upper ones.

Training-time augmentation (random flips, small rotations, RandomResizedCrop, ColorJitter), applied before the Normalize step, usually adds accuracy.

## 10. What is the IoU of two bounding boxes, and why train box regression with a GIoU or CIoU loss rather than with IoU itself?

IoU (intersection over union) = |P ∩ T| / |P ∪ T|: the overlap area of the predicted box P and the target box T, divided by the area of their union. It ranges from 0 (no overlap) to 1 (perfect match) and is the usual metric for evaluating predicted boxes (torchvision.ops.box_iou).

As a loss, IoU is 0 whenever the boxes don't overlap, however far apart they are, so its gradient is 0 and can't pull P toward T.

- GIoU = IoU − |S − (P ∪ T)| / |S|, where S is the smallest box enclosing both: the penalty grows as the boxes drift apart, giving a useful gradient. The loss is 1 − GIoU (torchvision.ops.generalized_box_iou_loss).
- CIoU also accounts for the distance between the box centers (relative to S's diagonal) and for how similar their aspect ratios are. Its loss, 1 − CIoU (torchvision.ops.complete_box_iou_loss), usually converges faster and gives more accurate boxes than MSE or GIoU.

## 11. Explain non-max suppression: why object detectors need it, and how the algorithm works.

A detector that makes predictions at many locations, such as a classifier slid across the image or a fully convolutional net that outputs a grid of boxes, typically detects the same object several times at slightly different positions. Non-max suppression keeps only the best box for each object:

1. Discard every box whose objectness score is below a threshold: the model believes there's no object there.
2. Take the remaining box with the highest objectness score, and discard all other remaining boxes that overlap it heavily (e.g., IoU above 0.6).
3. Repeat step 2 with the highest-scoring box that hasn't been kept or discarded yet, until none are left.

In TorchVision, torchvision.ops.nms(boxes, scores, iou_threshold) performs the overlap suppression and returns the indices of the kept boxes, sorted by decreasing score.

## 12. How is mean average precision (mAP) computed for an object detector, and what do mAP@0.5 and COCO's mAP@[.50:.95] mean?

Start from the precision/recall trade-off. For each recall level r = 0, 0.1, 0.2, …, 1.0, take the maximum precision the model achieves at a recall of at least r, then average these values: that's the average precision (AP) for one class. Taking the max at recall ≥ r means a stretch of the curve where precision rises again with recall isn't held against the model. The mAP is the mean of the per-class APs.

For detection, a prediction only counts as correct if its class is right and its box overlaps the ground-truth box enough: requiring an IoU greater than 0.5 gives mAP@0.5 (also written AP50), as in the PASCAL VOC challenge. COCO computes the mAP at IoU thresholds 0.50, 0.55, …, 0.95 and averages them, giving mAP@[.50:.95]. TorchMetrics implements this as MeanAveragePrecision.

## 13. How do transposed convolutions and skip connections help an FCN produce per-pixel predictions for semantic segmentation?

A pretrained CNN converted to an FCN typically has an overall stride of 32, so its final feature maps are 32 times smaller than the image: far too coarse to label pixels.

- A transposed convolutional layer (nn.ConvTranspose2d) upsamples: it's equivalent to stretching the input by inserting rows and columns of zeros, then running a regular convolution. Its stride sets how much the input is stretched, so a larger stride gives a larger output; e.g., kernel_size=4, stride=2, padding=1 exactly doubles the height and width. Unlike bilinear interpolation (fine only up to ×4 or ×8), it's trainable, so it can start close to linear interpolation and learn to do better.
- Upsampling ×32 in one step is still imprecise, so skip connections bring back detail from lower, higher-resolution layers: upsample ×2 and add the output of a lower layer with that resolution, upsample ×2 again and add an even lower layer's output, then upsample ×8.
