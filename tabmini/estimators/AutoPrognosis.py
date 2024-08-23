from pathlib import Path
import numpy as np
import pandas as pd
from autoprognosis.explorers.core.defaults import default_classifiers_names, default_regressors_names
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.studies.regression import RegressionStudy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class AutoPrognosis(BaseEstimator):
    """A scikit-learn compatible estimator that uses AutoPrognosis for both classification and regression."""

    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            model_names: list[str] = None,
            seed: int = 0,
            task: str = "classification",  # Task type: 'classification' or 'regression'
            kwargs: dict = {}
    ):
        self.path = path
        self.feature_names = []
        self.time_limit = time_limit
        self.task = task
        self.seed = seed
        self.kwargs = kwargs

        # Set default model names based on the task
        if self.task == "classification":
            self.model_names = model_names or default_classifiers_names
        elif self.task == "regression":
            self.model_names = model_names or default_regressors_names
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

        self.target_name = "target"

    def fit(self, X, y) -> 'AutoPrognosis':
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels for classification, real numbers for regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y, accept_sparse=True)

        self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_names.insert(0, self.target_name)

        train_data = pd.DataFrame(
            np.hstack((np.array(y)[:, np.newaxis], X)),
            columns=self.feature_names
        )

        if self.task == "classification":
            study = ClassifierStudy(
                train_data,
                target=self.target_name,
                workspace=self.path,
                score_threshold=0.3,
                classifiers=self.model_names,
                timeout=int(self.time_limit / len(self.model_names)),
                random_state=self.seed,
                **self.kwargs
            )
        elif self.task == "regression":
            # the only supported optimization metric for regression is r2
            study = RegressionStudy(
                train_data,
                target=self.target_name,
                workspace=self.path,
                score_threshold=15.0,
                regressors=self.model_names,
                timeout=int(self.time_limit / len(self.model_names)),
                random_state=self.seed,
                **self.kwargs
            )

        self.study_ = study.fit()

        return self

    def predict(self, X) -> np.ndarray:
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['study_'])
        return self.study_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification tasks.")
        
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ['study_'])

        probability_positive_class = self.study_.predict_proba(X).iloc[:, 1]
        probability_positive_class_scaled = (probability_positive_class - probability_positive_class.min()) / (
                probability_positive_class.max() - probability_positive_class.min() + 1e-10)

        # Create a 2D array with probabilities of both classes
        return np.vstack([1 - probability_positive_class_scaled, probability_positive_class_scaled]).T

    def decision_function(self, X):
        """
        Calculate decision function scores for classification tasks.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,)
            The decision function scores.
        """
        if self.task != "classification":
            raise AttributeError("decision_function is only available for classification tasks.")
        
        proba = self.predict_proba(X)
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))

        return decision
