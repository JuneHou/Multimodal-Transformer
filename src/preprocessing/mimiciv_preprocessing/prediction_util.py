from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

def run_xgb(x_train, y_train, x_test, gpu=0, seed=0, n_jobs=1):
    cv_folds = 5
    gs_metric = 'roc_auc'
    param_grid = {'max_depth': [5, 6, 7, 8],
                  'n_estimators': [200, 300],
                  'learning_rate': [0.3, 0.1, 0.05],
                  }
    
    est = xgb.XGBClassifier(verbosity=2, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), seed=seed,
                            tree_method='gpu_hist', gpu_id=gpu, eval_metric='logloss')
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator = est, param_grid=param_grid, scoring=gs_metric, cv= cv, n_jobs=n_jobs)
    gs.fit(x_train, y_train)

    y_pred_prob_train = gs.predict_proba(x_train)
    y_pred_train = gs.predict(x_train)

    y_pred_prob = gs.predict_proba(x_test)
    y_pred = gs.predict(x_test)

    return y_pred, y_pred_prob[:, 1], y_pred_train, y_pred_prob_train[:, 1], gs


def run_xgb_multilabel(x_train, y_train, x_test, gpu=0, seed=0, n_jobs=1):
    gs_metric = 'roc_auc_ovr'  # One-vs-Rest ROC AUC for multilabel
    param_grid = {'estimator__max_depth': [5, 6, 7, 8],
                  'estimator__n_estimators': [200, 300],
                  'estimator__learning_rate': [0.3, 0.1, 0.05],
                  }
    
    # MultiOutputClassifier wraps the XGBClassifier for multilabel prediction
    est = MultiOutputClassifier(xgb.XGBClassifier(verbosity=2, seed=seed,
                                                  tree_method='gpu_hist', gpu_id=gpu,
                                                  eval_metric='logloss'))

    gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=5, n_jobs=n_jobs)
    gs.fit(x_train, y_train)

    y_pred_prob_train = gs.predict_proba(x_train)
    y_pred_train = gs.predict(x_train)

    y_pred_prob = gs.predict_proba(x_test)
    y_pred = gs.predict(x_test)

    # Each column in the predictions corresponds to a label
    return y_pred, y_pred_prob, y_pred_train, y_pred_prob_train, gs


def run_xgb_multiclass(x_train, y_train, x_test, gpu=0, seed=0, n_jobs=1):
    num_classes = len(np.unique(y_train))
    cv_folds = 5
    gs_metric = 'accuracy'  # You can also use 'neg_log_loss' or 'roc_auc_ovr' if appropriate

    param_grid = {
        'max_depth': [5, 6, 7, 8],
        'n_estimators': [200, 300],
        'learning_rate': [0.3, 0.1, 0.05],
    }

    est = xgb.XGBClassifier(
        verbosity=1,
        objective='multi:softprob',
        num_class=num_classes,
        seed=seed,
        tree_method='gpu_hist',
        gpu_id=gpu,
        eval_metric='mlogloss',
        use_label_encoder=False  # Optional: suppress label encoder warning
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv, n_jobs=n_jobs)
    gs.fit(x_train, y_train)

    y_pred_prob_train = gs.predict_proba(x_train)  # shape: (n_train, num_classes)
    y_pred_train = gs.predict(x_train)

    y_pred_prob = gs.predict_proba(x_test)         # shape: (n_test, num_classes)
    y_pred = gs.predict(x_test)

    # Return full probability distributions instead of just one class
    return y_pred, y_pred_prob, y_pred_train, y_pred_prob_train, gs