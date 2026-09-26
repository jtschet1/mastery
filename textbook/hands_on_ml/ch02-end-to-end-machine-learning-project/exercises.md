# Chapter 2 — End-to-End Machine Learning Project

Exercises from *Hands-On Machine Learning with Scikit-Learn and PyTorch*
(Aurélien Géron). This chapter's exercises are all hands-on coding, so it
has no flashcards: do them in a notebook. The author's solutions are at the
end of `handson-mlp-main/02_end_to_end_machine_learning_project.ipynb`.

### Hands-on exercises

Summarized here; see the book for the full text.

- 1. Try an SVR (sklearn.svm.SVR) with linear and RBF kernels and various C
  and gamma values. How does the best SVR perform?
- 2. Replace GridSearchCV with RandomizedSearchCV.
- 3. Add a SelectFromModel transformer to the preparation pipeline to keep
  only the most important attributes.
- 4. Write a custom transformer that fits a KNeighborsRegressor on
  latitude/longitude to predict median income, and add its output to the
  preprocessing pipeline as a feature.
- 5. Use RandomizedSearchCV to explore preparation options automatically.
- 6. Reimplement StandardScalerClone from scratch, adding
  inverse_transform(), feature_names_in_, and get_feature_names_out().
- 7. Tackle a regression task of your choice end to end (e.g., the Vehicle
  or Bike Sharing dataset). Open-ended; no author solution.
