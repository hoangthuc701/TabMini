from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


import tabmini
import pandas as pd

working_directory = Path.cwd() / "workdir"

# Name of the method
method_name = "Ridge Regression"

# Define the pipeline
pipe = Pipeline(
    [
        ("scaling", MinMaxScaler()),  # Step 1: Scaling
        ("regress", Ridge(random_state=42)),  # Step 2: Ridge regression estimator
    ]
)

# Define the hyperparameter grid
param_grid = [
    {
        "regress__alpha": [0.5, 0.01, 0.002, 0.0004],  # Ridge regularization parameter
    }
]

# Perform grid search cross-validation
estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)


def formate_dataset(name, ds, target):
    dataset = {}
    ds['target'] = ds[target]
    dataset[name] = (ds.drop(columns=[target, 'target']), ds['target'])
    return dataset

fake_ds = pd.read_csv('ceramic_production_measurements_reduced.csv')

# load dataset 
dataset = formate_dataset('edgecurl' , fake_ds, 'EC-Messwert')

def run_experiment(framework, output_path, time_limit):
    # Load dataset
    print(framework)

    print(f'-------------------Time limit: {time_limit}-----------------------------------')
    test_scores, train_scores = tabmini.compare(
        method_name,
        estimator,
        dataset,
        working_directory,
        scoring_method="neg_mean_absolute_error",
        cv=3,
        time_limit=time_limit,
        framework=framework,
        device="cpu",
        n_jobs=-1,  # Time Limit does not play nice with threads
        task='regression',
    )

    test_scores.to_csv(Path(output_path ,f"results_{time_limit}.csv"), index_label="PMLB dataset")

run_experiment('autoprognosis', 'results', 60)