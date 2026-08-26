import warnings

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import (
    RandomForestClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

train = pd.read_csv("Datasets/Rachycentron/Train.csv")
test = pd.read_csv("Datasets/Rachycentron/Test.csv")

X_train = train.drop(columns=["label3"])
y_train = train["label3"]

X_test = test.drop(columns=["label3"])
y_test = test["label3"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__C": [0.1, 1, 10]
        }
    },

    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_jobs=-1))
        ]),
        "params": {
            "model__n_neighbors": [5, 11, 21],
            "model__weights": ["uniform", "distance"]
        }
    },

    "SVM": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                random_state=RANDOM_STATE
            ))
        ]),

        "params": [

            {
                "model__kernel": ["linear"],
                "model__C": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ]
            },

            {
                "model__kernel": ["rbf"],
                "model__C": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ],
                "model__gamma": [
                    "scale",
                    "auto",
                    0.1,
                    0.01,
                    0.001
                ]
            }

        ]

    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_samples=0.8
            ))
        ]),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_split": [2, 5]
        }
    },

    "XGBoost": {
        "pipeline": Pipeline([
            ("model", XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,


                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8
            ))
        ]),
        "params": {
            "model__learning_rate": [0.1, 0.05],
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5]
        }
    },

    "LightGBM": {
        "pipeline": Pipeline([
            ("model", LGBMClassifier(
                objective="multiclass",
                num_class=3,
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=1,
                max_depth=7,

                max_bin=63,
                subsample=0.8,
                subsample_freq=1
            ))
        ]),
        "params": {
            "model__learning_rate": [0.1, 0.05],
            "model__n_estimators": [100, 200],
            "model__num_leaves": [20, 31],
            "model__colsample_bytree": [0.7, 1.0]
        }
    },

    "CatBoost": {
        "pipeline": Pipeline([
            ("model", CatBoostClassifier(
                loss_function="MultiClass",
                random_state=RANDOM_STATE,
                verbose=False,
                thread_count=-1,

                bootstrap_type="Bernoulli",
                subsample=0.8
            ))
        ]),
        "params": {
            "model__learning_rate": [0.1, 0.05],
            "model__iterations": [100, 200],
            "model__depth": [4, 6]
        }
    }
}

results = []
grid_results = []

for name, cfg in models.items():

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    grid = GridSearchCV(
        estimator=cfg["pipeline"],
        param_grid=cfg["params"],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    test_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, test_pred)

    precision = precision_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    print("\nBest Parameters")
    print(grid.best_params_)

    print(f"\nBest CV Macro F1 : {grid.best_score_:.4f}")

    print("\nTest Metrics")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    results.append({

        "Model": name,

        "Best Parameters": str(grid.best_params_),

        "CV Macro F1": grid.best_score_,

        "Test Accuracy": accuracy,

        "Test Precision": precision,

        "Test Recall": recall,

        "Test Macro F1": macro_f1

    })

    cv_result = pd.DataFrame(grid.cv_results_)

    cv_result.insert(
        0,
        "Model",
        name
    )

    grid_results.append(
        cv_result
    )


summary = pd.DataFrame(results)

summary = summary.sort_values(
    by="Test Macro F1",
    ascending=False
).reset_index(drop=True)

grid_results = pd.concat(
    grid_results,
    ignore_index=True
)

print("\n")
print("=" * 120)
print("FINAL COMPARISON")
print("=" * 120)

print(summary)

with pd.ExcelWriter(
    "baseline_model_comparison_rachycentron.xlsx",
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    grid_results.to_excel(
        writer,
        sheet_name="GridSearch",
        index=False
    )
