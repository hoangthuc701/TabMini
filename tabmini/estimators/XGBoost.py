from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from xgboost import XGBClassifier, XGBRegressor


class XGBoost(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible estimator that uses XGBoost to fit and predict data for both classification and regression."""

    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            device: str = "cpu",
            seed: int = 0,
            task: str = "classification",  # Task type: 'classification' or 'regression'
            kwargs: dict = {}
    ):
        self.path = path
        self.time_limit = time_limit
        self.device = device
        self.seed = seed
        self.task = task
        self.kwargs = kwargs

        if self.task == "classification":
            self.model = XGBClassifier(
                n_estimators=2,
                max_depth=2,
                learning_rate=1,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=self.seed,
            )
            self.n_classes_ = 2
            self.classes_ = [0, 1]
        elif self.task == "regression":
            self.model = XGBRegressor(
                n_estimators=250, 
                max_depth=3, 
                eta=0.1, 
                subsample=0.7, 
                colsample_bytree=0.8,
                enable_categorical=True
                )
        else:
            raise ValueError("Task must be either 'classification' or 'regression'")

    def fit(self, X, y) -> 'XGBoost':
        X, y = check_X_y(X, y)

        self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_names.insert(0, "target")

        self.model.fit(X, y)

        return self

    def predict(self, X):
        check_is_fitted(self.model)
        # X = check_array(X)
        print('predicting xgboost')
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification tasks.")

        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        return self.model.predict_proba(X)

    def decision_function(self, X):
        if self.task != "classification":
            raise AttributeError("decision_function is only available for classification tasks.")

        proba = self.predict_proba(X)
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))

        return decision
