# Chapter 3 — Classification, to fall asleep to

Chapter three. Classification.

---

The two most common supervised learning tasks are regression, which predicts values, and classification, which predicts classes. Chapter two covered a regression task, predicting housing values. Chapter three covers classification systems.

---

The MNIST dataset. MNIST is a set of seventy thousand small images of digits handwritten by high school students and employees of the United States Census Bureau. Each image is labeled with the digit it represents. It has been studied so much that it is often called the hello world of machine learning: new classification algorithms are routinely tested on it, and anyone learning machine learning tackles it sooner or later.

Scikit-Learn has helper functions to download popular datasets, and MNIST is fetched from OpenML. Scikit-Learn's datasets package has three main kinds of functions. Fetch functions download real-life datasets. Load functions load small toy datasets bundled with Scikit-Learn. Make functions generate fake datasets, useful for tests. Generated datasets are usually returned as a pair of NumPy arrays, the inputs and the targets. Other datasets are returned as bunch objects, dictionaries whose entries can also be accessed as attributes, usually holding a description of the dataset, the input data, and the labels. The fetch openml function returns a Pandas data frame by default, which is not ideal for images, so its as frame option is set to false to get NumPy arrays instead.

There are seventy thousand images, each with seven hundred eighty-four features, because each image is twenty-eight by twenty-eight pixels and each feature is one pixel's intensity, from zero for white to two hundred fifty-five for black. Reshaping a feature vector into a twenty-eight by twenty-eight array and displaying it with Matplotlib shows the first image, which looks like a five, and its label is five. The labels are stored as strings.

The digits vary a lot in shape, but the images are clean, well centered, not too rotated, and roughly the same size, so the dataset needs little preprocessing. Real-world datasets are rarely this friendly.

A test set must be set aside before inspecting the data closely. MNIST, as returned by fetch openml, is already split: the first sixty thousand images form the training set and the last ten thousand the test set. The training set is already shuffled. Shuffling guarantees that all cross-validation folds are similar, with no fold missing some digits. It also helps algorithms that are sensitive to the order of training instances and perform poorly when they see many similar instances in a row. Shuffling can be a bad idea for time series data, such as stock prices or weather conditions.

---

Training a binary classifier. A simpler problem comes first: detecting a single digit, five. This five-detector is a binary classifier, distinguishing between just two classes, five and non-five. Its target vectors are true for every five and false for every other digit.

A good first classifier is Scikit-Learn's stochastic gradient descent classifier, or SGD classifier. It handles very large datasets efficiently, partly because SGD deals with training instances independently, one at a time, which also makes it well suited to online learning. Trained on the whole training set, it predicts that the first image is a five, which is correct.

---

Measuring accuracy using cross-validation. Evaluating a classifier is often significantly trickier than evaluating a regressor. Scikit-Learn's cross val score function, with three-fold cross-validation and accuracy as the scoring, gives the SGD classifier an accuracy, the ratio of correct predictions, above ninety-five percent on every fold.

A dummy classifier that puts every image in the most frequent class, here non-five, gets over ninety percent accuracy. Only about ten percent of the images are fives, so always guessing not-five is right about ninety percent of the time.

This is why accuracy is generally not the preferred performance measure for classifiers, especially with skewed datasets, where some classes are much more frequent than others. A much better way to evaluate a classifier is to look at its confusion matrix.

Cross-validation can also be implemented by hand, for more control. Scikit-Learn's stratified k-fold class produces folds that contain a representative ratio of each class. At each iteration, the classifier is cloned, the clone is trained on the training folds and makes predictions on the test fold, and the ratio of correct predictions is computed. This gives the same results as the cross val score function.

---

Confusion matrices. A confusion matrix counts the number of times instances of class A are classified as class B, for every pair of classes. For example, the number of times images of eights were classified as zeros is in row eight, column zero.

The matrix needs predictions to compare with the actual targets. The test set must stay untouched until the very end of the project, so the cross val predict function is used instead. Like cross val score, it performs k-fold cross-validation, but instead of scores it returns the predictions made on each test fold. This gives a clean prediction for every training instance, meaning an out-of-sample prediction, made by a model that never saw that instance during training. The confusion matrix function then compares the targets with these predictions.

Each row of a confusion matrix represents an actual class, and each column a predicted class. In the five-detector's matrix, the first row covers non-five images, the negative class. Fifty-three thousand eight hundred ninety-two were correctly classified as non-fives; these are true negatives. Six hundred eighty-seven were wrongly classified as fives; these are false positives, also called type one errors. The second row covers fives, the positive class. One thousand eight hundred ninety-one were wrongly classified as non-fives; these are false negatives, also called type two errors. Three thousand five hundred thirty were correctly classified as fives; these are true positives. A perfect classifier would have only true positives and true negatives, so its confusion matrix would have nonzero values only on its main diagonal, from top left to bottom right.

---

Precision and recall. A more concise metric than the full confusion matrix is the accuracy of the positive predictions, called precision. Precision is the number of true positives divided by the total number of positive predictions, that is, true positives plus false positives.

A classifier that makes only a single positive prediction, on the instance it is most sure of, and gets it right, has one hundred percent precision, but it ignores all the other positive instances. So precision is used together with recall, also called sensitivity or the true positive rate, TPR. Recall is the ratio of positive instances that the classifier correctly detects: true positives divided by true positives plus false negatives.

The five-detector has a precision of eighty-three point seven percent: when it claims an image is a five, it is right only that often. Its recall is sixty-five point one percent: it detects only that share of the fives. It looks much less impressive than its accuracy suggested.

Precision and recall are often combined into a single metric called the F1 score, which is convenient for comparing two classifiers. The F1 score is the harmonic mean of precision and recall. A regular mean treats all values equally, but the harmonic mean gives much more weight to low values, so a classifier only gets a high F1 score if both recall and precision are high. The F1 score equals two times precision times recall, divided by precision plus recall. Equivalently, it is the number of true positives divided by the sum of the true positives and half of the false negatives and false positives combined. The five-detector's F1 score is about zero point seven three.

The F1 score favors classifiers with similar precision and recall, which is not always what is needed. A classifier that detects videos safe for kids should prefer high precision with low recall: it may reject many good videos, but it keeps only safe ones, rather than letting a few really bad videos through. A human review step could even be added. A classifier that detects shoplifters in surveillance images can accept thirty percent precision if it has ninety-nine percent recall: the guards get some false alerts, but almost all shoplifters are caught. Medical diagnosis usually requires high recall to avoid missing anything important, since follow-up tests can rule out false positives.

Increasing precision reduces recall, and vice versa. This is called the precision-recall trade-off.

---

The precision-recall trade-off. For each instance, the SGD classifier computes a score with a decision function. If the score is greater than a threshold, it assigns the instance to the positive class; otherwise, to the negative class.

The book's example ranks a few digits by score. With the threshold in the middle, there are four true positives above it, actual fives, and one false positive, actually a six, so precision is eighty percent, four out of five. Of the six actual fives, four are detected, so recall is sixty-seven percent. Raising the threshold turns the six into a true negative, raising precision to one hundred percent, but one true positive becomes a false negative, lowering recall to fifty percent. Lowering the threshold increases recall and reduces precision.

The classifier's decision function method returns each instance's score, and any threshold can be applied to it. The SGD classifier's own threshold is zero. The first image, a five, scores about two thousand one hundred sixty-four, so it is detected at threshold zero, but missed at a threshold of three thousand. Raising the threshold decreases recall.

To choose a threshold, the cross val predict function is used again, this time returning the decision scores of all training instances rather than predictions. The precision recall curve function then computes the precision and recall for every possible threshold. Plotting both against the threshold shows that at three thousand, precision is near ninety percent and recall around fifty percent.

The precision curve is bumpier than the recall curve. Precision can sometimes go down when the threshold rises, although in general it goes up: in the ranked digits example, moving the middle threshold one digit to the right lowers precision from four out of five to three out of four, from eighty to seventy-five percent. Recall can only go down when the threshold rises, so its curve is smooth.

Another way to choose is to plot precision directly against recall. Precision starts to fall sharply at around eighty percent recall, so a good trade-off is probably just before that drop, for example at around sixty percent recall, depending on the project.

To get ninety percent precision, the lowest threshold that gives at least ninety percent precision is found with NumPy's arg max method, which returns the index of the first maximum value, here the first threshold that meets the condition. That threshold is about three thousand three hundred seventy. Comparing the scores with this threshold, instead of calling the predict method, gives a classifier with ninety percent precision, but only about forty-eight percent recall.

A classifier with virtually any precision is easy to build: just set a high enough threshold. But a high-precision classifier is not very useful if its recall is too low, and for many applications forty-eight percent recall would not be acceptable. When someone asks for ninety-nine percent precision, the right question is: at what recall?

Since Scikit-Learn version one point five, two classes make adjusting the threshold easier. The fixed threshold classifier wraps a binary classifier with a manually chosen threshold. If the classifier can estimate class probabilities, the threshold is a probability between zero and one, by default zero point five; otherwise it is a decision score, by default zero. The tuned threshold classifier uses k-fold cross-validation to find the best threshold for a given metric, by default balanced accuracy, the average of each class's recall.

---

The ROC curve. The receiver operating characteristic curve, or ROC curve, is another common tool for binary classifiers. It plots the true positive rate, another name for recall, against the false positive rate, FPR. The false positive rate, also called the fall-out, is the ratio of negative instances incorrectly classified as positive. It equals one minus the true negative rate, the ratio of negative instances correctly classified as negative, which is also called specificity. So the ROC curve plots sensitivity against one minus specificity.

Scikit-Learn's ROC curve function computes both rates for various thresholds. There is again a trade-off: the higher the recall, the more false positives the classifier produces. A purely random classifier's ROC curve is the diagonal line, and a good classifier stays as far from it as possible, toward the top-left corner.

Classifiers can be compared by the area under the curve, or AUC. A perfect classifier has a ROC AUC of one, and a purely random classifier has a ROC AUC of zero point five. The SGD five-detector's ROC AUC is about zero point nine six.

As a rule of thumb, the precision-recall curve, or PR curve, is preferable when the positive class is rare, or when false positives matter more than false negatives. Otherwise, the ROC curve is a good choice. The five-detector's ROC curve and ROC AUC look very good, but mostly because there are few positives compared with negatives. Its PR curve shows there is room for improvement, since the curve could be much closer to the top-right corner.

---

Comparing with a random forest. A random forest classifier can be compared with the SGD classifier through its PR curve and F1 score. The precision recall curve function needs a score for each instance, but a random forest classifier has no decision function method, because of the way it works. Instead, it has a predict proba method that returns estimated class probabilities for each instance, and the probability of the positive class serves as the score. Scikit-Learn classifiers always have a decision function method, a predict proba method, or both.

The cross val predict function, with its method set to predict proba, returns cross-validated class probabilities for every image. The model gives the first image an eighty-nine percent probability of being positive, and the second a ninety-nine percent probability of being negative. Each image is either positive or negative, so the probabilities in each row add up to one hundred percent.

These are estimated probabilities, not actual probabilities. Among images the model classified as positive with an estimated probability between fifty and sixty percent, about ninety-four percent are actually positive, so those estimates are much too low. Models can also be overconfident. Scikit-Learn's calibrated classifier class calibrates estimated probabilities with cross-validation, bringing them much closer to actual probabilities. This matters for tasks such as medical diagnosis, financial risk assessment, and fraud detection.

The random forest's PR curve is much better than the SGD classifier's, coming much closer to the top-right corner. With a probability threshold of fifty percent, its F1 score is about zero point nine three and its ROC AUC about zero point nine nine eight, both significantly better. Its precision is about ninety-nine percent and its recall about eighty-seven percent.

---

Multiclass classification. Multiclass classifiers, also called multinomial classifiers, distinguish between more than two classes. Some Scikit-Learn classifiers handle multiple classes natively, such as logistic regression, the random forest classifier, and Gaussian naive Bayes. Others are strictly binary, such as the SGD classifier and the support vector machine classifier. But multiple binary classifiers can be combined to classify more than two classes.

One strategy, for classifying digits into ten classes, is to train ten binary classifiers, one per digit: a zero-detector, a one-detector, and so on. To classify an image, each classifier outputs a decision score, and the class whose classifier gives the highest score wins. This is the one-versus-the-rest strategy, sometimes called one-versus-all.

Another strategy trains a binary classifier for every pair of digits: zeros against ones, zeros against twos, ones against twos, and so on. This is the one-versus-one strategy. With N classes, it needs N times N minus one, divided by two, classifiers, which for MNIST means forty-five. To classify an image, it runs all forty-five classifiers and picks the class that wins the most duels. Its main advantage is that each classifier trains only on the part of the training set containing its two classes.

Some algorithms, such as support vector machine classifiers, scale poorly with the size of the training set. For them, one-versus-one is preferred, because training many classifiers on small training sets is faster than training a few classifiers on large ones. For most binary classifiers, one-versus-the-rest is preferred.

Scikit-Learn detects when a binary classification algorithm is used for a multiclass task, and automatically runs one-versus-the-rest or one-versus-one, depending on the algorithm. A support vector machine classifier, trained on the first two thousand images with the original targets from zero to nine, uses one-versus-one and trains forty-five binary classifiers. It correctly predicts that the first image is a five, by making forty-five predictions and selecting the class that won the most duels.

Its decision function returns ten scores per instance, one per class. Each class's score is its number of won duels, plus or minus a small tweak of at most zero point three three, based on the classifier scores, to break ties. The highest score, about nine point three, belongs to class five. A trained classifier stores its list of target classes, sorted by value, in its classes attribute. For MNIST, each class's index happens to match the class itself, but in general the class label must be looked up in that list.

To force one-versus-one or one-versus-the-rest, Scikit-Learn provides the one versus one classifier and the one versus rest classifier, which wrap any classifier passed to them, even one that is not binary. A one-versus-the-rest wrapper around a support vector machine classifier trains ten estimators.

An SGD classifier trained on all ten classes uses one-versus-the-rest internally, training ten binary classifiers, and its decision function returns one score per class. It predicts that the first image is a three, which is wrong. Almost all its scores are strongly negative; class three scores about one thousand eight hundred twenty-four, and class five is not far behind at about minus one thousand three hundred eighty-six, so the classifier is not very confident.

The classes are roughly balanced, so accuracy is a reasonable metric here. The SGD classifier gets over eighty-five point eight percent on all cross-validation folds. A random classifier would get ten percent, so this is not bad, but it can be improved: simply scaling the inputs with a standard scaler raises accuracy above eighty-nine point one percent.

---

Error analysis. In a real project, the next steps would follow the project checklist: explore data preparation options, try multiple models, shortlist the best ones, fine-tune their hyperparameters with grid search, and automate as much as possible. Once a promising model is found, one way to improve it is to analyze the types of errors it makes.

The starting point is the confusion matrix, built from the predictions of the cross val predict function. With ten classes it contains many numbers, so a colored diagram is easier to read. Scikit-Learn's confusion matrix display class can plot one directly from the labels and predictions.

Most images lie on the main diagonal, meaning they were classified correctly. The diagonal cell for fives is slightly darker than the others, which could mean the model made more errors on fives, or that there are fewer fives in the dataset. So the matrix should be normalized, by dividing each value by the total number of images in its true class, that is, by its row's sum. The display can show the normalized values as percentages.

The normalized matrix shows that only eighty-two percent of fives were classified correctly. The most common error on fives was classifying them as eights, which happened to ten percent of fives. But only two percent of eights were classified as fives. Confusion matrices are generally not symmetrical.

To make the errors stand out, the correct predictions can be given zero weight. In this errors-only matrix, the column for class eight is very bright: many images were misclassified as eights, and this is the most common misclassification for almost all classes. These percentages must be read carefully, since correct predictions are excluded. Thirty-six percent in row seven, column nine means that thirty-six percent of the errors on sevens were misclassifications as nines, not that thirty-six percent of all sevens were. In reality, only three percent of sevens were misclassified as nines.

The errors-only matrix can also be normalized by column instead of by row. Then, for example, among the images wrongly classified as sevens, fifty-six percent are actually nines.

Here, the analysis suggests focusing on reducing the false eights. One option is to gather more training data for digits that look like eights but are not, so the classifier learns to tell them apart from real eights. Another is to engineer new features, such as an algorithm that counts closed loops: an eight has two, a six has one, and a five has none. Another is to preprocess the images, with Scikit-Image, Pillow, or OpenCV, to make patterns such as closed loops stand out more.

Analyzing individual errors also shows what the classifier is doing and why it fails. Images of threes and fives can be laid out like a confusion matrix: threes classified as threes, threes classified as fives, fives classified as threes, and fives classified as fives. Some of the misclassified digits are so badly written that even a human would have trouble classifying them, but most look like obvious errors. The human brain is a fantastic pattern recognition system, and the visual system does a lot of complex preprocessing before information reaches consciousness, so a task that feels simple is not necessarily simple.

The SGD classifier is a linear model. It assigns a weight per class to each pixel, and for a new image it sums the weighted pixel intensities to get a score for each class. Threes and fives differ by only a few pixels, so this model easily confuses them. The main difference is the position of the small line that joins the top line to the bottom arc. A three drawn with that junction slightly shifted to the left may be classified as a five, and vice versa. So this classifier is quite sensitive to image shifting and rotation.

One way to reduce the confusion between threes and fives is to preprocess the images so they are well centered and not too rotated, but that requires predicting the correct rotation of each image, which is not easy. A much simpler approach is to augment the training set with slightly shifted and rotated variants of the training images, which forces the model to tolerate such variations. This is called data augmentation.

---

Multilabel classification. So far each instance has been assigned to exactly one class. Sometimes a classifier should output multiple classes for each instance. A face-recognition classifier trained to recognize Alice, Bob, and Charlie should, given a picture of Alice and Charlie, output true, false, true: Alice yes, Bob no, Charlie yes. A classification system that outputs multiple binary tags is called a multilabel classification system.

The book's example gives each digit image two target labels: whether the digit is large, meaning seven, eight, or nine, and whether it is odd. A k-nearest neighbors classifier, which supports multilabel classification, unlike some classifiers, is trained on both targets. For the first image, a five, it correctly outputs not large and odd.

There are many ways to evaluate a multilabel classifier, and the right metric depends on the project. One approach is to measure the F1 score, or any other binary classifier metric, for each label and average the scores. The average F1 score across both labels is about zero point nine eight. This assumes all labels are equally important, which may not be true. If there are many more pictures of Alice than of Bob or Charlie, the score on Alice may deserve more weight. A simple option is to weight each label by its support, the number of instances with that target label.

A classifier that does not natively support multilabel classification, such as a support vector machine classifier, can be used by training one model per label. But this may fail to capture dependencies between the labels. A large digit is twice as likely to be odd as even, but the classifier for the odd label does not know what the classifier for the large label predicted. The solution is to organize the models in a chain: each model uses the input features plus the predictions of all the models before it in the chain.

Scikit-Learn's classifier chain class does this. By default, it trains each model with the true labels of the models before it in the chain. If its cross-validation hyperparameter is set, it instead uses cross-validation to get clean, out-of-sample predictions from each trained model for every training instance, and uses those predictions to train the later models. The order of the classifiers in the chain can affect the final performance. A chain of support vector machine classifiers, trained with three-fold cross-validation on the first two thousand images, also predicts not large and odd for the first image.

---

Multioutput classification. The last type of classification task is multioutput-multiclass classification, or simply multioutput classification. It generalizes multilabel classification so that each label can be multiclass, meaning it can have more than two possible values.

The example is a system that removes noise from images. It takes a noisy digit image as input and outputs a clean digit image, represented as an array of pixel intensities, like the MNIST images. The output is multilabel, with one label per pixel, and each label can take many values, since pixel intensity ranges from zero to two hundred fifty-five. So it is a multioutput classification system.

The line between classification and regression is sometimes blurry, as here: predicting pixel intensity is arguably closer to regression than to classification. Multioutput systems are not limited to classification, and a system could even output multiple labels per instance, mixing class labels and value labels.

The training and test inputs are created by adding random integer noise, between zero and one hundred, to the pixel intensities of the MNIST images, and the targets are the original images. A k-nearest neighbors classifier is trained on the noisy images and their clean targets. Given a noisy image from the test set, it outputs a cleaned-up image close to the target.

---

Summary.

Classification predicts classes. MNIST, seventy thousand labeled images of handwritten digits, is the standard first classification dataset. It comes already split into a training set and a test set, with the training set shuffled.

A binary classifier distinguishes two classes. Accuracy is misleading on skewed datasets: a classifier that never predicts five is ninety percent accurate on the five-detector task. The confusion matrix counts true negatives, false positives, false negatives, and true positives.

Precision is the share of positive predictions that are correct. Recall is the share of positive instances that are detected. The F1 score is their harmonic mean, and it is high only when both are high. Raising the decision threshold generally increases precision and decreases recall, so the threshold is chosen for the project's needs, from cross-validated decision scores and precision-recall curves.

The ROC curve plots the true positive rate against the false positive rate, and the area under it summarizes a classifier in one number. The precision-recall curve is preferable when positives are rare, or when false positives matter more. On the five-detector, a random forest clearly beats the SGD classifier.

Multiclass classification uses natively multiclass algorithms, or combines binary classifiers with the one-versus-the-rest or one-versus-one strategy. Scaling the inputs improves the SGD classifier's accuracy.

Error analysis with normalized confusion matrices shows which classes get confused, here mostly with eights and between threes and fives, and suggests fixes such as more data, new features, image preprocessing, and data augmentation.

Multilabel classification outputs several binary labels per instance, and classifier chains capture the dependencies between labels. Multioutput classification allows each label to take more than two values, as in removing noise from images.
