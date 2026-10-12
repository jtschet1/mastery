# Chapter 12 — Deep Computer Vision Using Convolutional Neural Networks, to fall asleep to

Chapter twelve. Deep computer vision using convolutional neural networks.

---

Perception is not trivial. IBM's Deep Blue beat the chess world champion Garry Kasparov in 1996, but only recently have computers been able to reliably detect a puppy in a picture or recognize spoken words. These tasks feel effortless to humans because perception happens mostly outside consciousness, in specialized sensory modules of the brain. By the time information reaches consciousness, it already carries high-level features, so subjective experience says little about how perception works.

Convolutional neural networks, or CNNs, emerged from the study of the brain's visual cortex and have been used in image recognition since the 1980s. Over the last fifteen years, more computing power, more data, and better training techniques have brought them to superhuman performance on some complex visual tasks. They power image search, self-driving cars, and video classification, and also work for voice recognition and natural language processing.

---

The architecture of the visual cortex. In experiments on cats in 1958 and 1959, and later on monkeys, David Hubel and Torsten Wiesel showed that many neurons in the visual cortex have a small local receptive field: they react only to stimuli in a limited region of the visual field. The receptive fields of different neurons overlap, and together they tile the whole visual field. The work earned them the Nobel Prize in Physiology or Medicine in 1981.

Some neurons react only to lines of one orientation. Others have larger receptive fields and react to more complex patterns that combine the lower-level ones, which suggested that higher-level neurons build on the outputs of neighboring lower-level neurons.

These studies inspired the neocognitron in 1980, which gradually evolved into CNNs. In 1998, Yann LeCun and colleagues introduced the LeNet-5 architecture, which banks widely used to recognize handwritten digits on checks. Besides fully connected layers and sigmoid activations, it introduced two new building blocks: convolutional layers and pooling layers.

A plain deep network of fully connected layers works for small images such as Fashion MNIST, but breaks down for larger ones because of the number of parameters. A one hundred by one hundred pixel image has ten thousand pixels, so a first layer of just one thousand neurons already needs ten million connections. CNNs avoid this with partially connected layers and weight sharing.

---

Convolutional layers. In a convolutional layer, each neuron is connected only to the pixels in its receptive field, not to every pixel of the image. Each neuron in the next convolutional layer is connected only to neurons within a small rectangle of the previous layer. The first hidden layer concentrates on small low-level features, the next assembles them into larger higher-level features, and so on. This hierarchy suits real-world images, which are full of composite objects, and it is one reason CNNs work so well for image recognition. Each layer is two-dimensional, so images need not be flattened.

A neuron in row i, column j is connected to a rectangle of neurons in the previous layer that starts at row i, column j and has the height and width of the receptive field. To give a layer the same height and width as the previous one, zeros are added around the inputs. This is called zero padding.

Receptive fields can also be spaced out, connecting a large input layer to a much smaller layer, which dramatically reduces computation. The step from one receptive field to the next is called the stride. With a stride of two, the receptive field of the neuron in row i, column j starts at twice its row index and twice its column index. The stride is usually the same in both directions.

---

Filters. A neuron's weights can be pictured as a small image the size of its receptive field, called a filter, a convolution kernel, or just a kernel. A seven by seven filter full of zeros except for a central column of ones makes a neuron ignore everything in its receptive field except the central vertical line. If every neuron in a layer uses this filter and the same bias, the layer's output enhances the vertical lines in the image and blurs the rest. A filter with a central row of ones does the same for horizontal lines.

A layer of neurons sharing one filter outputs a feature map, which highlights the areas of the image that activate the filter most. Filters are not designed by hand: during training, the layer learns the most useful filters for its task, and the layers above learn to combine them into more complex patterns. A single model that goes from raw inputs to final outputs like this is called end-to-end learning.

---

Stacking multiple feature maps. A convolutional layer has many filters and outputs one feature map per filter, so its output is three-dimensional. All neurons within a feature map share the same kernel and bias term, and different feature maps use different ones. A neuron's receptive field extends across all the feature maps of the previous layer. So a convolutional layer applies several trainable filters at once, and can detect several features anywhere in its input.

Weight sharing dramatically reduces the number of parameters. It also means a pattern learned in one location can be recognized in any location, while a fully connected network that learns a pattern in one location can only recognize it there.

Input images also have several channels: usually three, for red, green, and blue, called RGB; one for grayscale images; and sometimes many more, as in satellite images that capture infrared.

The output of a convolutional neuron is the bias of its feature map plus the weighted sum of all the inputs in its receptive field, across every row, every column, and every feature map of the previous layer. The bias works like a knob for the feature map's overall brightness. Neurons at the same position in different feature maps of a layer all receive exactly the same inputs.

---

Implementing convolutional layers with PyTorch. The book loads two sample images from Scikit-Learn, a Chinese tower and a dahlia flower, as a float tensor with pixel values from zero to one. PyTorch expects the channel dimension before the height and width, so the dimensions are permuted, and the images are center-cropped to seventy by one hundred twenty pixels.

PyTorch's two-dimensional convolution layer, with three input channels and thirty-two filters of seven by seven, turns each image into thirty-two feature maps. Two-dimensional refers to height and width, but the layer takes four-dimensional input: batch size, channels, height, and width.

The output is six pixels smaller in each direction, because by default the layer uses no padding, losing three pixels on each side with a seven by seven kernel. No padding is confusingly called valid padding, since every receptive field stays within valid positions of the input. Same padding adds enough zeros for the output to keep the input's size. A stride of two, with a padding of three, halves the output in each direction. PyTorch refuses same padding with a stride greater than one, since keeping the full size would drown the image in zeros.

The layer's weight tensor has the shape output channels, input channels, kernel height, kernel width, and its bias has one value per output channel. The image's height and width do not appear, because all neurons in a feature map share the same weights. So the layer accepts images of any size, as long as they are at least as large as the kernel and have the right number of channels.

A convolutional layer performs a linear operation, so each one must be followed by an activation function; otherwise a stack of them is equivalent to a single layer. The default initialization is uniform, between minus and plus one over the square root of the fan-in, which is the kernel height times the kernel width times the number of input channels. It is generally better to reinitialize the weights to suit the activation function, for example with He initialization for ReLU, and to set the biases to zero.

The main hyperparameters are the number of filters, the kernel size, the padding, the stride, and the activation function. Tuning them by cross-validation is very time-consuming, so well-known architectures serve as a guide.

---

Pooling layers. A pooling layer shrinks its input to reduce computation, memory use, and the number of parameters, which also limits overfitting. Each pooling neuron covers a small receptive field, with a size, stride, and padding type, but it has no weights or biases. It just aggregates its inputs, usually with the max or the mean.

Max pooling is the most common type. With a two by two kernel and a stride of two, only the largest of each group of four values passes to the next layer, so the height and width are halved. A pooling layer works on each channel independently, so the number of channels does not change.

Max pooling also introduces some invariance to small translations: shifting an image by one pixel may leave the output unchanged. Inserting a max pooling layer every few layers gives some translation invariance at a larger scale, plus a little rotational and scale invariance. This helps when the prediction should not depend on such details, as in classification.

But max pooling is very destructive: even a two by two kernel with a stride of two drops seventy-five percent of the input values. And some tasks need equivariance rather than invariance. In semantic segmentation, if the input shifts by one pixel, the output should shift by one pixel too.

---

Implementing pooling layers with PyTorch. PyTorch's max pooling layer takes a kernel size; the stride defaults to the kernel size, with no padding. The average pooling layer computes the mean instead. Average pooling used to be popular, but max pooling generally performs better. The mean loses less information, but the max keeps only the strongest features, giving the next layers a cleaner signal. Max pooling also gives stronger translation invariance and needs slightly less compute.

Pooling can also run along the depth dimension instead of the spatial dimensions. A CNN could learn several filters, each detecting a different rotation of the same pattern, such as a handwritten digit, and a depthwise max pooling layer would make the output the same whatever the rotation. The same idea applies to thickness, brightness, skew, or color. PyTorch has no such layer, but a custom module can rearrange the tensor so that the channels form the last dimension, apply one-dimensional max pooling, and rearrange it back.

Global average pooling computes the mean of each entire feature map, so it outputs one number per feature map and per instance. It is extremely destructive, but useful just before the output layer. PyTorch's adaptive average pooling layer, with an output size of one, does this without knowing the input size in advance, and so does simply taking the mean over the height and width.

---

CNN architectures. A typical CNN stacks a few convolutional layers, each followed by ReLU, then a pooling layer, then a few more convolutional layers and another pooling layer, and so on. The image gets smaller as it moves through the network, but also deeper, with more feature maps. On top sits a regular feedforward network of a few fully connected layers, ending in an output layer, such as a softmax for class probabilities.

Two stacked convolutional layers with three by three kernels are generally better than one layer with a five by five kernel: fewer parameters, less computation, and usually better performance. The exception is the first layer, which often has a large kernel, such as seven by seven, usually with a stride of two or more. This shrinks the image without losing much information, and with only three input channels it is not costly.

The book's CNN for Fashion MNIST starts with sixty-four seven by seven filters and a max pooling layer, then twice repeats a block of two three by three convolutional layers followed by max pooling. The number of filters doubles after each pooling layer, from sixty-four to one hundred twenty-eight to two hundred fifty-six. There are few kinds of low-level features but many ways to combine them, and since pooling halves each spatial dimension, the next layer can afford twice as many feature maps. A flatten layer then feeds two dense hidden layers with fifty percent dropout, and a ten-unit output layer that outputs logits for the cross-entropy loss.

The first dense layer needs two thousand three hundred four inputs: three by three pixels left after three pooling layers, times two hundred fifty-six feature maps. PyTorch's lazy linear layer can instead infer its input size on its first call. This model reaches about ninety-two percent test accuracy, better than the earlier dense networks.

---

The ImageNet challenge. Progress in CNNs is well measured by the ImageNet Large Scale Visual Recognition Challenge, or ILSVRC. In six years, the top-five error rate fell from over twenty-six percent to under two point three percent. The top-five error rate is the fraction of test images whose correct class is not among the system's five most confident predictions. There are one thousand classes, some very subtle, including one hundred twenty dog breeds.

---

LeNet-5. Yann LeCun's LeNet-5, from 1998, is perhaps the best-known CNN architecture, widely used for handwritten digit recognition on MNIST. On thirty-two by thirty-two inputs, it alternates three convolutional layers with five by five kernels and two average pooling layers, then has a fully connected layer of eighty-four units and a ten-unit output layer. It looks much like the Fashion MNIST model. A modern version would use ReLU instead of tanh, and softmax instead of its radial basis function output.

---

AlexNet. AlexNet won the 2012 challenge by a large margin, with a top-five error rate of seventeen percent against twenty-six percent for the runner-up. Built by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, it is similar to LeNet-5 but much larger and deeper, and it was the first to stack convolutional layers directly on top of one another instead of following each one with a pooling layer. It has five convolutional layers, starting with eleven by eleven kernels, three max pooling layers, two fully connected layers of four thousand ninety-six units, and an output layer of one thousand units.

To reduce overfitting, the authors applied dropout at fifty percent to the two large fully connected layers, and used data augmentation: random shifts, horizontal flips, and lighting changes. They also used local response normalization, in which the most strongly activated neurons inhibit neurons at the same position in neighboring feature maps, pushing feature maps to specialize. Batch normalization has mostly replaced it. ZFNet, which won in 2013, is essentially AlexNet with a few tweaked hyperparameters.

---

Data augmentation. Data augmentation artificially enlarges the training set with realistic variants of each training instance, which reduces overfitting. Ideally a human could not tell which images were generated. White noise does not help, because the changes should be learnable. Typical augmentations shift, rotate, and resize images, adjust colors and contrast to simulate lighting, and flip images horizontally, except for text and other asymmetrical objects. TorchVision provides transforms for all of these.

Augmentation can also generate extra samples of rare classes in an unbalanced dataset; the book calls this the synthetic minority oversampling technique, or SMOTE. Test-time augmentation classifies several augmented versions of a test image and combines the predictions: if three say bus and seven say truck, it is probably a truck.

---

GoogLeNet. GoogLeNet, by Christian Szegedy and colleagues at Google Research, won the 2014 challenge with a top-five error rate below seven percent. It was much deeper than earlier CNNs, thanks to subnetworks called inception modules, named after the movie Inception. Inception modules use parameters so efficiently that GoogLeNet has about six million parameters, ten times fewer than AlexNet.

An inception module sends its input through four parallel branches: a one by one convolution; a one by one convolution followed by a three by three convolution; a one by one convolution followed by a five by five convolution; and a max pooling layer followed by a one by one convolution. The different kernel sizes capture patterns at different scales. Every layer uses a stride of one and same padding, so all branch outputs have the same height and width and can be concatenated along the depth dimension.

The one by one convolutions have three roles. They capture patterns across channels. They output fewer feature maps than they receive, acting as bottleneck layers that cut computation and parameters, which speeds up training and improves generalization. And each pair of a one by one layer followed by a larger layer acts like a single, more powerful convolutional layer: a convolutional layer is like sweeping a dense layer across the image, and such a pair is like sweeping a two-layer network. The whole module is a convolutional layer on steroids.

GoogLeNet is one tall stack. Its first layers quickly divide the height and width by four, starting with a large seven by seven kernel. The backbone is nine inception modules, with a couple of max pooling layers between them. A global average pooling layer then reduces the final seven by seven feature maps to one value each. Losing the spatial information is fine for classification, and it removes the need for several fully connected layers on top, which cuts parameters and overfitting. The network ends with dropout and a one thousand unit softmax output layer. Two auxiliary classifiers on intermediate modules, meant to fight vanishing gradients, turned out to matter little. Later variants include Inception-v3 and Inception-v4.

---

ResNet. Kaiming He and colleagues won the 2015 challenge with a residual network, or ResNet, with a top-five error rate under three point six percent. The winning variant had one hundred fifty-two layers. It confirmed the trend toward deeper models with fewer parameters.

The key to training such a deep network is skip connections, also called shortcut connections: the signal feeding into a layer is also added to the output of a layer higher up the stack. When the input is added to a network's output, the network learns the target function minus the input, instead of the target function itself. This is called residual learning.

A newly initialized network has weights close to zero, so it outputs values close to zero. With a skip connection, it starts out as the identity function, outputting a copy of its input, and since the target function is often close to the identity, training speeds up considerably. With many skip connections, the network makes progress even when some layers have not started learning, because the signal easily crosses the whole network. A deep residual network is a stack of residual units, each a small network with a skip connection.

ResNet starts and ends like GoogLeNet, without the dropout layer, and in between is a very deep stack of residual units. Each unit has two three by three convolutional layers with batch normalization and ReLU, which preserve the spatial dimensions. Every few units, the number of feature maps doubles while the height and width are halved by a convolution with a stride of two. The unit's input then has a different shape from its output, so the skip connection passes it through a one by one convolution with a stride of two and the right number of feature maps.

ResNet-34 has three residual units with sixty-four feature maps, four with one hundred twenty-eight, six with two hundred fifty-six, and three with five hundred twelve. Deeper versions such as ResNet-152 use bottleneck units of three layers: a one by one convolution that cuts the number of feature maps by four, a three by three convolution, and a one by one convolution that restores the original depth.

Skipping a random set of residual units for each training mini-batch, a technique called stochastic depth, speeds up training considerably without hurting accuracy. Google's Inception-v4 later merged the ideas of GoogLeNet and ResNet, reaching a top-five error rate close to three percent.

---

Xception. Xception, short for Extreme Inception, was proposed in 2016 by François Chollet, the author of Keras. It significantly outperformed Inception-v3 on a huge task with three hundred fifty million images and seventeen thousand classes. It merges the ideas of GoogLeNet and ResNet, but replaces inception modules with depthwise separable convolution layers.

A regular convolutional layer captures spatial patterns, such as an oval, and cross-channel patterns, such as a mouth, nose, and eyes forming a face, at the same time. A separable layer assumes the two can be modeled separately. It first applies a single spatial filter to each input channel, then a one by one convolution that looks only across channels. An inception module sits between these two extremes, which is why Xception counts as a GoogLeNet variant.

Since a separable layer has only one spatial filter per input channel, it should not follow a layer with few channels, such as the input. So Xception starts with two regular convolutional layers, then uses thirty-four separable ones, plus a few max pooling layers, global average pooling, and a dense output layer.

PyTorch has no separable convolution layer, but one is easy to build: a depthwise convolution, made by setting the convolution's groups argument and output channels equal to the input channels so that each channel gets its own filter, followed by a one by one convolution. Separable layers use fewer parameters, less memory, and less computation, and often perform better, so they make a good default, except after layers with few channels.

---

SENet. The Squeeze-and-Excitation Network, or SENet, won the 2017 challenge with a top-five error rate of two point two five percent. It boosts existing architectures, such as inception networks and ResNets, by adding a small network, called an SE block, to every inception module or residual unit.

An SE block looks only at the depth dimension of its unit's output and learns which features are usually active together. If it sees strong activation in the mouth and nose feature maps but only mild activation in the eye feature map, it boosts the eye feature map; more precisely, it scales down the irrelevant feature maps. This recalibration helps resolve ambiguities.

The block has three layers. Global average pooling outputs the mean activation of each feature map. A dense hidden layer with ReLU squeezes these values, typically into sixteen times fewer numbers, forming a low-dimensional embedding. A dense output layer with the sigmoid activation turns it into one score per feature map, between zero and one, and the feature maps are multiplied by these scores.

---

Other noteworthy architectures. VGGNet, by Karen Simonyan and Andrew Zisserman of Oxford's Visual Geometry Group, was the 2014 runner-up. It is a simple, classical stack of two or three convolutional layers and a pooling layer, repeated to sixteen or nineteen convolutional layers, with many small three by three filters, and a dense network on top.

ResNeXt replaces each residual unit's layers with many parallel stacks, such as thirty-two, of three convolutional layers, where the first two layers of each stack use only a few filters, so the parameter count matches ResNet. The stacks' outputs are added together.

DenseNet is built from dense blocks in which each layer receives the concatenated outputs of all earlier layers in the block. Blocks are separated by transition layers. It achieves excellent accuracy with comparatively few parameters.

MobileNets are lightweight, fast models for mobile and web applications, based on depthwise separable convolutions. SqueezeNet, ShuffleNet, and MNasNet are other mobile architectures. A Cross Stage Partial Network, or CSPNet, is like a DenseNet, except that part of each block's input bypasses the block and is concatenated directly to its output.

EfficientNet is arguably the most important of these. Its authors proposed a principled way to scale any CNN by jointly increasing its depth, its width, meaning the number of filters per layer, and its input resolution. This is called compound scaling. They used neural architecture search to find a good architecture for a scaled-down version of ImageNet, then scaled it up. EfficientNet models outperformed all existing models across all compute budgets, and EfficientNet V2, in 2021, improved training time and parameter efficiency further.

ConvNeXt is a ResNet with tweaks borrowed from vision transformers, such as large seven by seven kernels and fewer activation functions and normalization layers in each residual unit.

---

Compound scaling. The compute budget is measured on a logarithmic scale by a number called phi, which increases by one each time the budget doubles. Depth, width, and resolution scale as alpha, beta, and gamma raised to the power phi. These three factors must be greater than one, and alpha times beta squared times gamma squared should be close to two. For EfficientNet, a grid search on the small baseline model, EfficientNet B0, with phi fixed at one, gave alpha equal to one point two, and beta and gamma equal to one point one. Increasing phi then produced EfficientNet B1 through B7.

---

Choosing the right CNN architecture. The best architecture depends on what matters most: accuracy, model size, inference speed, or energy consumption. Among TorchVision's pretrained classifiers, larger models are generally more accurate, but not always. EfficientNet v2 small beats Inception v3 in both size and accuracy, though not in compute. AlexNet has sixty-one million parameters but only about fifty-seven percent top-one accuracy on ImageNet, while ResNet 152, of similar size, reaches eighty-two percent. EfficientNet v2 large is the most accurate in the book's list, at almost eighty-six percent.

---

GPU memory requirements: inference versus training. The weights of a two hundred million parameter model take only eight hundred megabytes, but running it typically needs about five gigabytes per image at inference, and more for training.

Consider one convolutional layer with two hundred five by five filters processing a one hundred fifty by one hundred RGB image. It has only fifteen thousand two hundred parameters, where a fully connected layer with outputs of the same size would need about one hundred thirty-five billion. But it performs two hundred twenty-five million multiplications, and its output takes twelve megabytes for a single image, or one point two gigabytes for a batch of one hundred.

At inference, a layer's memory can be released as soon as the next layer is computed, so only two consecutive layers need to fit. During training, everything computed in the forward pass must be kept for the backward pass, so the activations of all layers must fit at once.

If training runs out of memory, the options are a smaller batch size, possibly accumulating gradients over several batches before each update; larger strides; fewer layers; sixteen-bit floats; distributing the model across devices; or offloading memory-hungry modules to the CPU.

Activation checkpointing, also called gradient checkpointing, trades compute for memory: only some activations are saved in the forward pass, and the rest are recomputed during the backward pass from the nearest checkpoint. PyTorch's checkpoint function wraps a module call, with no change to the architecture. The forward pass must give the same result when run twice, so custom modules must avoid in-place operations and control their randomness.

---

Reversible residual networks. RevNets, proposed by Aidan Gomez and colleagues in 2017, avoid saving activations altogether, for only about thirty-three percent more compute. Each reversible layer takes two inputs of equal size. Its first output is the first input plus a function f of the second input. Its second output is the second input plus a function g of the first output. During backpropagation, the inputs are recovered from the outputs by subtraction: the second input is the second output minus g of the first output, and then the first input is the first output minus f of the second input.

The functions f and g can be anything that preserves the shape and is deterministic, such as a few convolutional layers with stride one and same padding. Layers that change the shape, such as strided convolutions, cannot be reversible, so their activations are saved, but a CNN needs only a handful of them. RevNets also underpin a transformer architecture called Reformer.

---

Implementing ResNet-34 with PyTorch. A residual unit module has a main path made of a three by three convolution, batch normalization, ReLU, a second three by three convolution, and batch normalization. When the unit's stride is greater than one, the skip connection is a one by one convolution with that stride, plus batch normalization; otherwise it is an identity module that returns its input unchanged. The unit adds the outputs of the two paths and applies ReLU.

The full ResNet-34 is one sequential stack: a seven by seven convolution with sixty-four filters and a stride of two, batch normalization, ReLU, and max pooling; then a loop that adds the sixteen residual units, with a stride of two whenever the number of filters changes; then adaptive average pooling, a flatten layer, and a lazy linear output layer. The 2015 winning architecture fits in about forty-five lines of code.

---

Using TorchVision's pretrained models. Standard models such as ResNet and ConvNeXt rarely need to be implemented by hand, since TorchVision provides pretrained versions, as do the TIMM library and the Hugging Face Hub.

The book loads the base variant of ConvNeXt, pretrained on ImageNet; TorchVision downloads and caches the weights. Images must be preprocessed exactly as during training, and the weights object provides the right transforms: bringing the images to two hundred twenty-four by two hundred twenty-four pixels, and standardizing each color channel with ImageNet's means and standard deviations. The model is switched to evaluation mode, with autograd off, before predicting.

The model outputs one thousand logits per image, one per ImageNet class. The tower is classified as a palace and the flower as a daisy, reasonable substitutes since ImageNet has no Chinese tower or dahlia class, and the tower is part of Beijing's Summer Palace.

---

Pretrained models for transfer learning. For classes that are not in ImageNet, the lower layers of a pretrained model can still be reused. The book fine-tunes ConvNeXt on the Flowers102 dataset, which has one hundred two classes but only ten training images per class. That is too few to train from scratch, but enough to exceed ninety percent accuracy with a good pretrained model.

The model's head ends with a linear layer that outputs one thousand logits, so it is replaced with one that outputs one hundred two. All parameters are frozen except those of the head. Training only the head for a few epochs already reaches about ninety percent accuracy. Then the whole model is unfrozen, and training continues with a learning rate typically ten times lower.

Data augmentation helps further: random horizontal flips, small random rotations, random resized crops, and random color changes, all applied before the ImageNet normalization step. TorchVision also has an auto augment transform, which applies several augmentation operations optimized for ImageNet.

Other options include ensembles, studying failure cases, unfreezing layers gradually from the top, and smaller learning rates for lower layers. It also pays to use models pretrained on similar images: TorchGeo for geospatial data, Project MONAI for medical images, and AgML for agricultural images.

---

Classification and localization. Localizing an object is a regression task: predicting four numbers, such as the center coordinates, width, and height of a bounding box. A second dense output head with four units is added, for example on top of the global average pooling layer, next to the classification head. Training uses the cross-entropy loss for the class and the mean squared error, or MSE, for the box, combined as a weighted sum.

The catch is that Flowers102 has no bounding boxes. Labeling is often one of the hardest and most costly parts of a project. Many annotation tools exist, such as Label Studio and Roboflow, often AI-assisted. Crowdsourcing platforms such as Amazon Mechanical Turk suit very large jobs but take real effort to manage. A few hundred or a couple of thousand images can be labeled by hand in a few days, which also builds understanding of the data.

TorchVision has a bounding box type that records the box format and the image size, and its version two transforms update bounding boxes along with the images when resizing or cropping. Rotation is the exception: a box cannot rotate, so it becomes the box that encloses the rotated box, which may be too large for the object. With the MSE, a ten-pixel error costs the same on a large box as on a small one; taking the square root of the width and height before computing the loss reduces this.

---

Intersection over union. The most common metric for bounding boxes is the intersection over union, or IoU, also called the Jaccard index. It is the area of overlap between the predicted and target boxes, divided by the area of their union. It ranges from zero, for no overlap, to one, for perfect overlap.

The IoU is a poor training loss, because it is zero whenever the boxes do not overlap, however far apart they are, so the gradient is zero too. The generalized IoU, or GIoU, from 2019, subtracts from the IoU the fraction of the smallest enclosing box that neither box covers. It gets smaller as the boxes move apart, so gradient descent can pull the prediction toward the target. The loss is one minus the GIoU.

The complete IoU, or CIoU, from 2020, also accounts for the distance between the box centers, relative to the diagonal of the enclosing box, and for the similarity of the aspect ratios. It usually converges faster and gives more accurate boxes than the MSE or the GIoU, and it is becoming the default localization loss. TorchVision implements all of these.

---

Object detection. Classifying and localizing multiple objects in an image is called object detection. It used to be done by sliding a CNN, trained to classify and locate one roughly centered object, across the image. Besides class probabilities and a box, the CNN predicted an objectness score: the probability that an object is centered in the region, a single sigmoid output trained with binary cross-entropy. A separate no-object class generally worked worse: whether an object is present and what kind it is are best answered separately.

In the book's example, an image of roses is cut into a five by seven grid, and the CNN slides over every three by three region, giving fifteen predictions. A box may extend beyond its region, since the CNN can guess where the unseen part of an object lies. Predictions with a low objectness score are ignored. Sliding over smaller and larger regions catches objects of other sizes.

The same object is often detected several times, so the duplicates are removed by non-max suppression. First, drop all boxes whose objectness score is below a threshold. Then find the remaining box with the highest objectness score, and drop the other boxes that overlap it heavily, for example with an IoU above sixty percent. Repeat until no more boxes can be dropped. This works well, but running the CNN many times is slow.

---

Fully convolutional networks. Fully convolutional networks, or FCNs, were introduced by Jonathan Long and colleagues in 2015, for semantic segmentation. They pointed out that the dense layers at the top of a CNN can be replaced by convolutional layers. A dense layer of two hundred neurons on top of one hundred seven by seven feature maps computes exactly the same numbers as a convolutional layer with two hundred seven by seven filters and valid padding, which outputs two hundred one by one feature maps. In general, the number of filters must equal the number of dense units, the kernel size must equal the size of the input feature maps, and the padding must be valid.

This matters because a dense layer needs a fixed input size, while a convolutional layer accepts images of any size, as long as the number of channels matches. A network of only convolutional and pooling layers can be trained and run on images of any size.

Suppose a flower CNN outputs one hundred seven values: one hundred two class logits, one objectness logit, and four box coordinates. Its dense layers can be converted just by copying their weights, without retraining. If its last convolutional layer outputs seven by seven feature maps for a two hundred twenty-four pixel image, a four hundred forty-eight pixel image gives fourteen by fourteen maps, and the converted output layer produces an eight by eight grid of predictions, each for one region of the image. This equals sliding the original CNN over sixty-four positions, but the network looks at the image only once.

---

You Only Look Once. YOLO, proposed by Joseph Redmon and colleagues in 2015, is a fast and accurate detection architecture that runs in real time on video. It works much like the FCN just described, with three differences.

First, each grid cell only handles objects whose bounding box center lies inside it. The center coordinates are relative to the cell, but the box can extend well beyond it. Second, each cell outputs two boxes, each with its own objectness score, to handle two objects centered in the same cell. Third, each cell outputs one class probability distribution, over the twenty classes of the PASCAL VOC dataset, rather than one per box. Classes are assigned to boxes in post-processing, by matching each box to the regions of this coarse class probability map. For a person standing in front of a car, the car's large box matches a region where the car class dominates, and the person's smaller box matches a region inside it where the person class dominates.

YOLO was first built in Darknet, Redmon's framework written in C, and soon ported to PyTorch. Redmon's team followed with YOLOv2, YOLOv3, and YOLO9000, and other teams have continued since 2020. YOLOv3 added anchor priors, exploiting the fact that box shapes depend on the class: people tend to have vertical boxes, cars usually do not. Later versions added more boxes per cell, many more classes, up to nine thousand for YOLO9000, and skip connections to recover spatial resolution. Tiny YOLO variants run at over one thousand frames per second, with slightly lower accuracy.

---

Mean average precision. Object detectors are usually evaluated with the mean average precision. A precision-recall curve can contain sections where precision rises as recall increases. If a model has ninety percent precision at ten percent recall but ninety-six percent at twenty percent recall, there is no reason to use it at ten percent recall. So for each recall level, from zero to one hundred percent in steps of ten, the maximum precision achievable at that recall or higher is taken, and these maximums are averaged. This is the average precision. Averaging it over all classes gives the mean average precision.

In detection, a prediction counts as correct only if its class is right and its IoU with the target exceeds a threshold. The PASCAL VOC challenge uses a threshold of zero point five. The COCO competition averages the mean average precision over thresholds from zero point five to zero point nine five, in steps of zero point zero five: a mean mean average precision.

---

Pretrained detection models. TorchVision has no YOLO model, but the Ultralytics library provides pretrained YOLO models, trained on the COCO dataset, whose detection annotations cover eighty classes, such as person, car, and bicycle. The book runs the medium variant of YOLOv9, which returns each detected object's class name, confidence, and box corners. Ultralytics can also train YOLO models on custom data.

TorchVision provides other pretrained detectors. Faster R-CNN has two stages: a CNN processes the image, then a region proposal network proposes the boxes most likely to contain an object, and a classifier runs on each proposed box. SSD is a single-stage detector, like YOLO, and SSDlite is a lightweight version for mobile devices. RetinaNet introduced the focal loss, which gives more weight to difficult samples and improves performance on small objects and rare classes. FCOS is a single-stage fully convolutional detector that predicts boxes directly, without anchor boxes.

---

Object tracking. In video, objects must also be tracked from frame to frame, even as they move, change size and appearance, or are temporarily hidden. DeepSORT, a popular tracker, combines classical algorithms with deep learning. Kalman filters predict each tracked object's current position from its past detections, assuming roughly constant speed. A deep learning model measures how much each new detection resembles each tracked object. The Hungarian algorithm then matches new detections to tracked objects, minimizing both the distance to the predicted positions and the difference in appearance.

When a red ball bounces off a blue ball, the Kalman filter expects them to pass through each other. Matching on position alone would swap the balls' identities, but the appearance measure keeps each detection matched to the right ball. Ultralytics supports tracking, using BoT-SORT by default, a faster and more accurate relative of DeepSORT with camera-motion compensation.

---

Semantic segmentation. Semantic segmentation classifies every pixel by the class of the object it belongs to, such as road, car, or pedestrian, without separating different objects of the same class. It allows, for example, removing the background behind a person in a video call. The main difficulty is that layers with strides greater than one make a regular CNN lose spatial resolution, so it may know that a person is somewhere in the bottom left of the image, but not much more precisely.

The approach of Long and colleagues converts a pretrained CNN to an FCN and upsamples its output, which is thirty-two times smaller than the input. Bilinear interpolation works only up to about four or eight times, so they used a transposed convolutional layer. It is equivalent to stretching the image by inserting rows and columns of zeros, then applying a regular convolution. It can start out close to linear interpolation, then learn to do better. In a transposed convolution, a larger stride gives a larger output. It is sometimes called a deconvolution, but that name is mathematically wrong.

---

Other convolutional layers. One-dimensional convolutional layers process sequences, such as time series or text. Three-dimensional ones process volumes, such as 3D PET scans. Setting the dilation of any convolutional layer to two or more gives an à trous convolutional layer, French for "with holes". Its filter is spread out by inserting zeros between its weights, which enlarges the receptive field at no extra computational cost and with no extra parameters.

---

Skip connections and instance segmentation. Upsampling in a single step was still too imprecise, so Long and colleagues added skip connections from lower layers, which still have a higher resolution. In their best architecture, the output is upsampled by two, the output of a lower layer at that scale is added, the result is upsampled by two again, the output of an even lower layer is added, and a final step upsamples by eight. Upsampling beyond the original image size increases its resolution, which is called super-resolution.

Instance segmentation also separates individual objects of the same class, such as each bicycle in a row of bicycles. Mask R-CNN, by Kaiming He and colleagues in 2017, extends Faster R-CNN by predicting a pixel mask for each bounding box. TorchVision includes pretrained FCN and Mask R-CNN models, and its transforms handle masks and videos as well as bounding boxes.

Since 2020, transformers have also entered computer vision. Active research areas include robustness to adversarial images, explainability, realistic image generation, single-shot learning, predicting the next frames of a video, and combining text and image tasks.

---

Summary.

CNNs were inspired by the visual cortex. A convolutional layer connects each neuron to a small receptive field, and all neurons in a feature map share one filter, so a pattern learned in one place is recognized anywhere, with far fewer parameters than a dense layer. Pooling layers have no weights; max pooling shrinks feature maps and adds some translation invariance, at the cost of discarding most values.

Typical CNNs alternate convolutional and pooling layers, shrinking the image while adding feature maps, then finish with dense layers. Each landmark architecture added an idea: AlexNet depth and stacked convolutional layers, GoogLeNet inception modules with one by one bottlenecks, ResNet skip connections, Xception separable convolutions, SENet feature map recalibration, EfficientNet compound scaling, and ConvNeXt transformer-inspired tweaks.

Training needs far more memory than inference, because activations are kept for the backward pass; smaller batches, sixteen-bit floats, activation checkpointing, and reversible networks reduce it. Pretrained models can be adapted by replacing the head, training it with the rest frozen, then fine-tuning everything at a lower learning rate.

Localization adds a bounding box head, measured by intersection over union and trained with losses such as CIoU. Fully convolutional networks and YOLO detect multiple objects in one pass, non-max suppression removes duplicates, and mean average precision measures the result. Semantic segmentation classifies every pixel, recovering resolution with transposed convolutions and skip connections, and instance segmentation also separates individual objects.
