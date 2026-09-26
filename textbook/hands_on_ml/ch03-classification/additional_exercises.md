# Chapter 3 — Classification

Supplementary flashcards for *Hands-On Machine Learning with Scikit-Learn
and PyTorch* (Aurélien Géron), written by Claude from the book's text. They
are not the author's. The author's exercises for this chapter are all
hands-on coding (listed in exercises.md), so all 20 of this chapter's cards
are here.

Each `##` heading is a flashcard front; the text under it is the back.
Run `python3 textbook/flashcards/build_cards.py` after editing.

## 1. How do you load MNIST with Scikit-Learn, and what do you get back?

mnist = fetch_openml("mnist_784", as_frame=False) downloads it from OpenML.org (and caches it). The sklearn.datasets package has fetch_* functions for real datasets, load_* functions for small bundled toy datasets, and make_* functions that generate fake data. Fetched datasets come back as a Bunch, a dictionary whose entries are also attributes (data, target, DESCR). fetch_openml() returns a DataFrame and a Series by default; as_frame=False gives NumPy arrays, which suit images better.

mnist.data has shape (70000, 784): each row is a 28 × 28 image flattened into pixel intensities from 0 (white) to 255 (black). Gotcha: the labels are strings, so a 5-detector's target is y_train == "5", not == 5. The first 60,000 images form the training set, already shuffled, and the last 10,000 the test set. Shuffling keeps cross-validation folds similar and helps algorithms that struggle with many similar instances in a row, but it's a bad idea for time series.

## 2. Why can a 5-detector with over 95% cross-validated accuracy still be a poor classifier?

Only about 10% of MNIST images are 5s, so a DummyClassifier that always predicts the most frequent class ("not 5") scores about 91% accuracy without ever detecting a 5. On skewed datasets, where some classes are much more frequent than others, accuracy mostly reflects the class balance rather than how well the model finds the rare class. Compare against such a baseline, and judge the model with metrics focused on the positive class: the confusion matrix, precision and recall, and PR or ROC curves. Accuracy is fine when classes are balanced, as with the ten digits in the multiclass task.

## 3. Sketch stratified k-fold cross-validation implemented by hand, and explain the role of each piece.

When you need more control than cross_val_score() offers, write the loop yourself:

```python
skfolds = StratifiedKFold(n_splits=3)  # shuffle=True if not already shuffled
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    clone_clf.fit(X_train[train_index], y_train_5[train_index])
    y_pred = clone_clf.predict(X_train[test_index])
    n_correct = sum(y_pred == y_train_5[test_index])
    print(n_correct / len(y_pred))  # accuracy on this fold
```

StratifiedKFold produces folds that each keep a representative ratio of every class, and its split() method yields index arrays rather than data. clone() (from sklearn.base) creates a fresh, unfitted copy with the same hyperparameters, so each fold trains from scratch and the original estimator stays untouched. Each iteration trains on the other folds, predicts the held-out fold, and prints that fold's accuracy.

## 4. How do you get a confusion matrix without touching the test set, and how do you read a binary one?

cross_val_predict() performs k-fold cross-validation like cross_val_score(), but returns predictions instead of scores: each training instance is predicted by a model that never saw it during training, so every prediction is clean (out-of-sample). Then call confusion_matrix(y_train_5, y_train_pred).

Each row is an actual class and each column a predicted class:
- First row, actual negatives (non-5s): true negatives, then false positives (type I errors).
- Second row, actual positives (5s): false negatives (type II errors), then true positives.

A perfect classifier has nonzero counts only on the main diagonal. The SGD 5-detector's matrix was [[53892, 687], [1891, 3530]].

## 5. Define precision and recall, and explain why you need both.

precision = TP / (TP + FP): the fraction of positive predictions that are correct.

recall = TP / (TP + FN): the fraction of actual positives that the classifier detects. It's also called sensitivity or the true positive rate (TPR).

Each is easy to max out on its own: a classifier that makes a single, very confident positive prediction that happens to be right has 100% precision while missing almost every positive, and one that flags everything as positive has 100% recall. The SGD 5-detector's 95% accuracy hid 83.7% precision and 65.1% recall: when it claims an image is a 5 it's right 83.7% of the time, and it finds only 65.1% of the 5s. Compute them with precision_score(y_true, y_pred) and recall_score(y_true, y_pred).

## 6. What is the F1 score, why is it a harmonic mean, and when is it the wrong metric to optimize?

F1 = 2 / (1/precision + 1/recall) = 2 × precision × recall / (precision + recall) = TP / (TP + (FN + FP)/2). A harmonic mean gives much more weight to low values than a regular mean, so F1 is only high when both precision and recall are high, which makes it a convenient single number for comparing classifiers: f1_score(y_true, y_pred).

But F1 favors classifiers whose precision and recall are similar, and many applications care much more about one of them:
- Precision first: a classifier that picks videos safe for kids should reject many good videos (low recall) rather than let a few really bad ones through.
- Recall first: a shoplifter detector with 30% precision is fine if it has 99% recall, since guards can dismiss false alerts; medical diagnosis also favors recall, with follow-up tests ruling out false positives.

## 7. Explain the precision/recall trade-off in terms of decision scores and the decision threshold.

A classifier such as SGDClassifier computes a score for each instance with its decision function (decision_function()) and predicts positive when the score exceeds a threshold (0 for SGDClassifier's predict()). Picture the instances sorted by score: raising the threshold turns some false positives into true negatives, which usually increases precision, but also turns some true positives into false negatives, which decreases recall. Lowering it does the opposite.

Recall can only go down as the threshold rises, so its curve is smooth. Precision usually goes up but can dip: if raising the threshold drops a true positive while a false positive stays above it, precision can fall from 4/5 to 3/4. That's why the precision curve looks bumpier. Neither improves for free, so you choose the threshold that fits your application.

## 8. Walk through choosing a decision threshold that achieves 90% precision, in code.

Get out-of-sample decision scores instead of predictions, compute precision and recall for every threshold, and take the lowest threshold that reaches 90% precision:

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
idx = (precisions >= 0.90).argmax()  # index of the first True
threshold_90 = thresholds[idx]
y_train_pred_90 = (y_scores >= threshold_90)
```

argmax() on a boolean array returns the first index of the maximum, i.e., the first True. precision_recall_curve() appends a final precision of 1 and recall of 0 (an infinite threshold), so precisions and recalls have one more element than thresholds; drop it with [:-1] when plotting them against thresholds. Then check what it costs: here recall fell to about 48%. If someone asks for 99% precision, ask "at what recall?"

## 9. What do FixedThresholdClassifier and TunedThresholdClassifierCV do, and why use them rather than thresholding scores by hand?

Added in Scikit-Learn 1.5 (in sklearn.model_selection), both wrap a binary classifier so the threshold becomes part of the model: predict() applies it, and the wrapper can be saved, cross-validated, or put in a pipeline like any other classifier, with no custom thresholding code around it.
- FixedThresholdClassifier lets you set the threshold yourself, e.g., FixedThresholdClassifier(sgd_clf, threshold=3370.0). If the wrapped classifier has predict_proba(), the threshold is a probability between 0 and 1 (default 0.5); otherwise it's a decision score comparable to decision_function() outputs (default 0).
- TunedThresholdClassifierCV uses k-fold cross-validation to find the threshold that optimizes a metric, by default balanced accuracy (the average of each class's recall). Set scoring to optimize another metric, and read the chosen value from best_threshold_.

## 10. What does a ROC curve plot, and how do you compute it and summarize it with a single number?

It plots the true positive rate (recall, or sensitivity) against the false positive rate for every possible threshold. FPR = FP / (FP + TN) is the fraction of negatives wrongly flagged as positive (the fall-out); it equals 1 − specificity, where specificity, the true negative rate, is TN / (TN + FP). So a ROC curve is sensitivity versus 1 − specificity.

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores) computes the points (thresholds in decreasing order). It's a trade-off again: the higher the recall, the more false positives. A purely random classifier traces the diagonal; a good one stays as far from it as possible, toward the top-left corner. The area under the curve (AUC) summarizes it: roc_auc_score(y_train_5, y_scores) is 1 for a perfect classifier and 0.5 for a random one.

## 11. When should you prefer the precision/recall curve over the ROC curve, and why?

Rule of thumb: use the PR curve when the positive class is rare or when you care more about false positives than false negatives; otherwise use the ROC curve.

The reason is the denominators. The FPR divides false positives by the number of actual negatives, so when negatives vastly outnumber positives, even many false positives barely move it, and the ROC curve and its AUC look excellent. Precision divides by the number of positive predictions, so those same false positives show up clearly. For the 5-detector (only about 10% positives), the SGD model's ROC AUC of 0.96 looked great, while its PR curve showed plenty of room for improvement: it could get much closer to the top-right corner.

## 12. How do you get scores for PR and ROC curves from a classifier with no decision_function(), such as RandomForestClassifier?

Scikit-Learn classifiers always have decision_function() or predict_proba(), sometimes both. predict_proba() returns one column per class (in the order of classes_), each row summing to 1, and the positive class's column works as a score:

```python
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # estimated probability of a 5
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)
```

The thresholds are now probabilities, so for instance y_scores_forest >= 0.5 predicts positive whenever the estimated probability is at least 50%. The random forest clearly beat the SGD classifier: its PR curve came much closer to the top-right corner, with an F1 score of about 0.93 and a ROC AUC of about 0.998.

## 13. Are the outputs of predict_proba() real probabilities, and what can you do if they aren't?

They're estimates and can be miscalibrated. Among the images the random forest gave a 50–60% probability of being a 5, about 94% actually were 5s, so those estimates were far too low; other models are overconfident instead. Well calibrated means that among the instances given a probability p of being positive, about a fraction p really are. CalibratedClassifierCV (in sklearn.calibration) wraps a classifier and uses cross-validation to calibrate its estimated probabilities, bringing them much closer to actual probabilities. That matters when decisions rely on the probability values themselves rather than just the ranking or the predicted class, as in medical diagnosis, financial risk assessment, or fraud detection.

## 14. Compare the one-versus-the-rest and one-versus-one strategies for multiclass classification with binary classifiers.

- One-versus-the-rest (OvR, also one-versus-all): train one binary classifier per class (a 0-detector, a 1-detector, and so on), N in total, and predict the class whose classifier outputs the highest decision score.
- One-versus-one (OvO): train one classifier per pair of classes, N × (N − 1) / 2 in total (45 for the 10 digits), run the instance through all of them, and predict the class that wins the most duels.

OvO trains many more classifiers, but each one only on the instances of its two classes. That suits algorithms that scale poorly with the size of the training set, such as support vector machines, because training many classifiers on small sets is faster than training a few on large ones. For most binary classifiers, OvR is preferred. Classifiers such as LogisticRegression, RandomForestClassifier, and GaussianNB handle multiple classes natively and need neither.

## 15. What happens when you fit a binary-only classifier such as SVC or SGDClassifier on a 10-class target, and how do you inspect or override the strategy?

Scikit-Learn detects the multiclass target and applies OvR or OvO automatically: SVC uses OvO (45 binary classifiers for the digits), SGDClassifier uses OvR (10). decision_function() then returns one score per class for each instance; for SVC, a class's score is its number of won duels plus a small tie-breaking adjustment based on the classifiers' scores. Map the best column back to a label through classes_, which lists the classes sorted by value (index and label only coincide by luck, as with digits):

```python
scores = svm_clf.decision_function([some_digit])  # shape (1, 10)
svm_clf.classes_[scores.argmax()]  # '5'
```

To force a strategy, wrap any classifier in OneVsRestClassifier or OneVsOneClassifier from sklearn.multiclass, e.g., OneVsRestClassifier(SVC(random_state=42)); after fitting, its estimators_ attribute holds the underlying binary classifiers (10 for OvR on the digits).

## 16. How do you use a confusion matrix plot for multiclass error analysis, and what do its normalization options reveal?

Get clean predictions with cross_val_predict(), then call ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred):
- normalize="true" divides each row by the number of instances of that actual class, so cells show fractions of each class; raw counts would confuse "more errors" with "fewer instances". values_format=".0%" displays percentages.
- sample_weight=(y_train_pred != y_train) gives correct predictions zero weight, so only errors show. With normalize="true", a cell is then the share of that class's errors going to each predicted class; normalize="pred" normalizes by column instead, showing which actual classes each class's wrong predictions came from.

Confusion matrices are usually asymmetric: 10% of 5s were predicted as 8s, but only 2% of 8s as 5s. Many digits were wrongly predicted as 8s, which suggests gathering more images that look like 8s but aren't, engineering features such as the number of closed loops, or preprocessing images to make such patterns stand out.

## 17. Why does the SGD classifier confuse 3s and 5s, and what are two ways to reduce that confusion?

SGDClassifier is a linear model: it assigns a weight per class to each pixel and scores an image by summing its weighted pixel intensities. 3s and 5s differ by only a few pixels, mainly the position of the small stroke that joins the top line to the bottom arc, so a slightly shifted or rotated digit can tip the scores toward the other class. The model is quite sensitive to image shifts and rotations.

Two remedies:
- Preprocess the images so they're well centered and not too rotated, which is hard because you must estimate each image's rotation.
- Data augmentation: add slightly shifted and rotated copies of the training images, forcing the model to tolerate such variations. It's much simpler.

## 18. Show how to train and evaluate a multilabel classifier in Scikit-Learn.

Build a 2D target with one column per binary label, and use a classifier that supports multilabel classification natively (not all do), such as KNeighborsClassifier:

```python
y_train_large = (y_train >= "7")
y_train_odd = (y_train.astype("int8") % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])  # array([[False, True]]): not large, odd
```

The right metric depends on the project. A common approach computes a binary metric for each label and averages it: f1_score(y_multilabel, y_train_knn_pred, average="macro"), with predictions from cross_val_predict(), treats all labels as equally important, while average="weighted" weights each label by its support (the number of instances that have it), useful when, say, Alice appears in far more photos than Bob.

## 19. What problem does ClassifierChain solve for multilabel classification, and what does its cv argument change?

With a classifier that can't handle multilabel targets natively, like SVC, you can train one model per label, but independent models can't exploit dependencies between labels: a large digit (7, 8, or 9) is twice as likely to be odd as even, yet the "odd" model never sees what the "large" model predicted. ClassifierChain (in sklearn.multioutput) links the models: each one receives the input features plus the predictions of every model earlier in the chain.

By default, each model is trained using the true labels of the earlier positions. With cv set, e.g., ClassifierChain(SVC(), cv=3, random_state=42), it instead uses cross-validation to get clean, out-of-sample predictions from each trained model and trains the later models on those, which better matches what they'll receive at prediction time. The order of the chain can affect performance.

## 20. Distinguish binary, multiclass, multilabel, and multioutput classification, with an example of each.

- Binary: two classes, like 5 versus not-5.
- Multiclass (multinomial): one label with more than two possible classes, like which digit from 0 to 9.
- Multilabel: several binary labels per instance, like a face tagger that outputs [True, False, True] for "Alice yes, Bob no, Charlie yes".
- Multioutput (multioutput–multiclass): several labels per instance, each of which can take more than two values. Example: a denoiser that takes a noisy digit image and outputs the clean image, one label per pixel with intensities from 0 to 255; a KNeighborsClassifier trained with noisy images as inputs and the originals as targets can do it.

The line between classification and regression blurs here: predicting pixel intensities is arguably closer to regression, and a multioutput system can even mix class labels and numeric labels for the same instance.
