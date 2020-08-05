import os
import pickle
import logging

import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, precision_recall_curve, average_precision_score
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import matplotlib.pyplot as plt

from config import PREPROCESSED_DATA_DIR, MODELS_DIR

logging.basicConfig(filename=os.path.join(MODELS_DIR, 'train.log'),
                    filemode='a', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def calculate_boosting(params: dict) -> dict:
    logging.info(f"Training with params: {params}")
    num_round = params.get('n_estimators', 1000)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=0,
                          early_stopping_rounds=10)

    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)

    y_val = dvalid.get_label()
    roc_auc = roc_auc_score(y_val, predictions)
    prec_recall_score = average_precision_score(y_val, predictions)

    predictions = np.where(predictions > 0.5, 1, 0)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    logging.info('Roc_auc {:.3f}, Precision {:.3f}'.format(roc_auc, precision))
    logging.info('Recall {:.3f}, Prec_recall_auc {:.3f}, Num trees: {}'.format(recall,
                                                                               prec_recall_score,
                                                                               gbm_model.best_iteration + 1))
    params['n_estimators'] = gbm_model.best_iteration + 1

    loss = 1 - precision
    return {'loss': loss, 'status': STATUS_OK, 'model': gbm_model}


def optimize(number_steps: int = 200) -> dict:
    space = {
        'n_estimators': 1000,
        'eta': 0.1,
        'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.3, 1, 0.02),
        'gamma': hp.quniform('gamma', 0.3, 1, 0.02),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1, 0.02),
        'eval_metric': 'map',
        'objective': 'binary:logistic',
        'nthread': 3,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'verbosity': 0,
        'seed': 1
    }
    best = fmin(calculate_boosting,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=number_steps)
    return best


def evaluate_model(gbm_model, watch_list, thres=0.5):
    num_trees = gbm_model.best_iteration + 1
    for data, descr in watch_list:
        predictions = gbm_model.predict(data,
                                        ntree_limit=num_trees)

        y_val = data.get_label()
        roc_auc = roc_auc_score(y_val, predictions)
        prec_recall_score = average_precision_score(y_val, predictions)

        predictions = np.where(predictions > thres, 1, 0)
        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        logging.info(f'Metrics with thres {thres} {descr}: ')
        logging.info('Roc_auc {:.3f}, Precision {:.3f}'.format(roc_auc, precision))
        logging.info('Recall {:.3f}, Prec_recall_auc {:.3f} Num_trees: {}'.format(recall,
                                                                                  prec_recall_score,
                                                                                  num_trees))


def plot_precision_recall_curve(model: xgb.core.Booster,
                                d_matrix_data: xgb.core.DMatrix,
                                model_name_prefix: str):
    predictions = model.predict(d_matrix_data)
    y_test = d_matrix_data.get_label()
    prec, recall, thres = precision_recall_curve(y_test,
                                                 predictions)

    proper_thres = (recall > 0.7) & (prec > 0.8)

    min_thres = np.where(proper_thres)[0][0]
    max_thres = np.where(proper_thres)[0][-1]

    y = [precision_score(y_test, np.where(predictions > thres[max_thres - 1], 1, 0)),
         precision_score(y_test, np.where(predictions > thres[min_thres - 1], 1, 0))]
    x = [recall_score(y_test, np.where(predictions > thres[max_thres - 1], 1, 0)),
         recall_score(y_test, np.where(predictions > thres[min_thres - 1], 1, 0))]

    thres_list = [thres[max_thres - 1], thres[min_thres - 1]]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(recall, prec, c='r', alpha=0.5, linewidth=3)
    ax.plot(recall[proper_thres], prec[proper_thres], linewidth=5, c='g', label='proper thresholds')

    ax.scatter(x, y, s=100, c='g')

    for x_i, y_i, thres_i in zip(x, y, thres_list):
        ax.annotate('thres {:.2f}: recall {:.2f}, precision {:.2f}'.format(thres_i, x_i, y_i), (x_i, y_i),
                    xytext=(x_i - 0.45, y_i - 0.02), fontsize=12)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    path_img = os.path.join(MODELS_DIR, f'precision_recall_curve_{model_name_prefix}.png')
    plt.savefig(path_img)
    logging.info(f'Precision-recall curve saved to {path_img}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_steps_opt', '--num_steps_opt',
                        help='num steps in search params', default=300, type=int)
    args = parser.parse_args()

    logging.info(f'Train xgboost with paraneters optimization in {args.num_steps_opt} steps')

    train_data = np.load(os.path.join(PREPROCESSED_DATA_DIR, 'train.npz'), allow_pickle=True)
    x_train, y_train, feature_names = train_data['arr_0'], train_data['arr_1'], train_data['arr_2']

    val_data = np.load(os.path.join(PREPROCESSED_DATA_DIR, 'val.npz'), allow_pickle=True)
    x_val, y_val = val_data['arr_0'], val_data['arr_1']

    test_data = np.load(os.path.join(PREPROCESSED_DATA_DIR, 'test.npz'), allow_pickle=True)
    x_test, y_test = test_data['arr_0'], test_data['arr_1']

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dvalid = xgb.DMatrix(x_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)

    trials = Trials()
    best_hyperparams = optimize(number_steps=args.num_steps_opt)
    logging.info("The best hyperparameters are: ")

    best_hyperparams['max_depth'] += 1
    model_prefix = '_'.join(['{}_{}'.format(param, value) for param, value
                             in best_hyperparams.items()])
    best_hyperparams.update({'eval_metric': ['logloss', 'map'],
                             'objective': 'binary:logistic',
                             'nthread': 3,
                             'booster': 'gbtree',
                             'tree_method': 'exact',
                             'verbosity': 0,
                             'seed': 1,
                             'n_estimators': 1000,
                             'eta': 0.1})

    result = calculate_boosting(best_hyperparams)

    watch_list = [(dtrain, 'train'), (dvalid, 'eval'), (dtest, 'test')]
    evaluate_model(result['model'], watch_list=watch_list)

    xgb.plot_importance(result['model'])
    path_feat_import = os.path.join(MODELS_DIR, f'feature_importance_{model_prefix}.png')
    plt.savefig(path_feat_import, bbox_inches="tight")

    logging.info(f'Feature importance saved to {path_feat_import}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'model_{model_prefix}.bin')
    pickle.dump(result['model'], open(model_path, 'wb'))
    logging.info(f'Final model saved to {model_path}')

    plot_precision_recall_curve(result['model'],
                                dtest,
                                model_name_prefix=model_prefix)
