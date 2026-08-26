import warnings

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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


RULES = {
    "Salinity": {
        "HS": [(25, 30, True, True)],
        "S": [(22, 25, True, False), (30, 32, False, True)],
    },
    "pH": {
        "HS": [(7.8, 8.2, True, True)],
        "S": [(7.5, 7.8, True, False), (8.2, 8.6, False, True)],
    },
    "Sea water temperature": {
        "HS": [(24, 30, True, True)],
        "S": [(20, 24, True, False), (30, 34, False, True)],
    },
    "DO": {
        "HS": [(6, 8, False, False)],
        "S": [(4, 6, True, True)],
    },
    "TSS": {
        "HS": [(0, 20, False, False)],
        "S": [(20, 50, True, True)],
    },
    "Ammonium (NH₄⁺-N)": {
        "HS": [(0, 0.05, False, False)],
        "S": [(0.05, 0.10, True, True)],
    },
    "Phosphate (PO₄³⁻-P)": {
        "HS": [(0, 0.10, False, False)],
        "S": [(0.10, 0.20, True, True)],
    },
    "Fluoride (F⁻)": {
        "HS": [(0, 1.0, False, False)],
        "S": [(1.0, 1.5, False, True)],
    },
    "Cyanide (CN⁻)": {
        "HS": [(0, 0.005, False, False)],
        "S": [(0.005, 0.01, True, True)],
    },
    "As": {
        "HS": [(0, 0.002, False, False)],
        "S": [(0.002, 0.005, True, True)],
    },
    "Cd": {
        "HS": [(0, 0.001, False, False)],
        "S": [(0.001, 0.005, True, True)],
    },
    "Pb": {
        "HS": [(0, 0.02, False, False)],
        "S": [(0.02, 0.05, True, True)],
    },
    "Cr(VI)": {
        "HS": [(0, 0.01, False, False)],
        "S": [(0.01, 0.02, True, True)],
    },
    "Total Chromium": {
        "HS": [(0, 0.05, False, False)],
        "S": [(0.05, 0.10, True, True)],
    },
    "Cu": {
        "HS": [(0, 0.10, False, False)],
        "S": [(0.10, 0.20, True, True)],
    },
    "Zn": {
        "HS": [(0, 0.30, False, False)],
        "S": [(0.30, 0.50, True, True)],
    },
    "Mn": {
        "HS": [(0, 0.25, False, False)],
        "S": [(0.25, 0.50, True, True)],
    },
    "Fe": {
        "HS": [(0, 0.25, False, False)],
        "S": [(0.25, 0.50, True, True)],
    },
    "Oil and Grease": {
        "HS": [(0, 0.20, False, False)],
        "S": [(0.20, 0.50, True, True)],
    },
}

RANKS = {
    "Salinity": 3,
    "pH": 4,
    "Sea water temperature": 2,
    "DO": 1,
    "TSS": 8,
    "Ammonium (NH₄⁺-N)": 5,
    "Phosphate (PO₄³⁻-P)": 9,
    "Fluoride (F⁻)": 14,
    "Cyanide (CN⁻)": 6,
    "As": 12,
    "Cd": 13,
    "Pb": 16,
    "Cr(VI)": 11,
    "Total Chromium": 19,
    "Cu": 17,
    "Zn": 15,
    "Mn": 18,
    "Fe": 10,
    "Oil and Grease": 7
}



def calculate_weights(ranks_dict):
    inverse_ranks = {k: 1.0 / v for k, v in ranks_dict.items()}
    total_inv = sum(inverse_ranks.values())
    return {k: v / total_inv for k, v in inverse_ranks.items()}

WEIGHTS = calculate_weights(RANKS)


def get_interval_info(x, intervals):
    if not intervals:
        return False, np.inf, 1.0, 0, 1.0

    is_in = False
    min_dist = np.inf
    closest_width = 1.0
    center = 0
    radius = 1.0

    for lo, hi, lc, rc in intervals:
        left_cond = x >= lo if lc else x > lo
        right_cond = x <= hi if rc else x < hi

        if left_cond and right_cond:
            is_in = True
            min_dist = 0.0
            closest_width = hi - lo
            center = (hi + lo) / 2.0
            radius = (hi - lo) / 2.0
            break

        dist = lo - x if x < lo else (x - hi if x > hi else 0)

        if dist < min_dist:
            min_dist = dist
            closest_width = hi - lo
            center = (hi + lo) / 2.0
            radius = (hi - lo) / 2.0

    if closest_width <= 0: closest_width = 1e-6
    if radius <= 0: radius = 1e-6

    return is_in, min_dist, closest_width, center, radius


def calculate_suitability_score(x, hs_intervals, s_intervals, gamma_us=1.0):
    gamma_s = -np.log(0.675 / 0.875)

    in_hs, dist_to_hs, _, center_hs, radius_hs = get_interval_info(x, hs_intervals)

    if in_hs:
        score = 1.0 - 0.125 * (abs(x - center_hs) / radius_hs)
        return score

    in_s, dist_to_s, width_s, _, _ = get_interval_info(x, s_intervals)

    if in_s:
        score = 0.875 * np.exp(-gamma_s * (dist_to_hs / width_s))
        return score

    if not s_intervals:
        dist_to_boundary = dist_to_hs
        width_scaling = radius_hs * 2
    else:
        dist_to_boundary = dist_to_s
        width_scaling = width_s

    score = 0.675 * np.exp(-gamma_us * (dist_to_boundary / width_scaling))
    return score


def enrich_features_with_expert_knowledge(df, rules, weights, gamma_us=1.0):
    df_enhanced = df.copy()

    for col, rule in rules.items():
        if col in df.columns:
            hs_intervals = rule.get("hs", [])
            s_intervals = rule.get("s", [])
            w_i = weights.get(col, 0)

            scores = df[col].apply(
                lambda x: calculate_suitability_score(x, hs_intervals, s_intervals, gamma_us)
            )

            penalty_col_name = f"{col}_Risk"
            df_enhanced[penalty_col_name] = w_i * (1.0 - scores)

    return df_enhanced


train = pd.read_csv("Datasets/Rachycentron/Train.csv")
test = pd.read_csv("Datasets/Rachycentron/Test.csv")

X_train_raw = train.drop(columns=["label3"])
y_train = train["label3"]

X_test_raw = test.drop(columns=["label3"])
y_test = test["label3"]

X_train = enrich_features_with_expert_knowledge(X_train_raw, RULES, WEIGHTS, gamma_us=1.0)
X_test = enrich_features_with_expert_knowledge(X_test_raw, RULES, WEIGHTS, gamma_us=1.0)


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
    "enhanced_baseline_model_comparison_rachycentron.xlsx",
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
