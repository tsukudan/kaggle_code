import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import StreamHandler, Formatter ,FileHandler, getLogger,DEBUG, INFO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
import xgboost as xgb
from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = "result_tmp/"
SAMPLE_SABMIT_FILE = "../input/sample_submission.csv"

def gini(y,pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr,tpr) - 1
    return g


def gini_xgb(pred,y):
    y = y.get_label()
    return "gini",- gini(y,pred)


if __name__ == "__main__":
    
    logger.setLevel(DEBUG)
    log_fmt = Formatter("%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s")


    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(log_fmt)
    logger.addHandler(stream_handler)

    file_handler = FileHandler(DIR + "train.py.log" , "a")
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(log_fmt)
    logger.addHandler(file_handler)


    logger.info("start")

    df = load_train_data()

    x_train = df.drop("target", axis=1)
    y_train = df["target"].values

    use_cols = x_train.columns.values

    logger.debug("train columns: {} {}".format(use_cols.shape, use_cols))
    logger.info("data preparation end {}".format(x_train.shape))


    cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {"max_depth":[3,5,7],
                  "learning_rate":[0.1],
                  "min_child_weight":[3,5,7],
                  "n_estimators":[10000],
                  "colsample_bytree":[0.8,0.9],
                  "colsample_bylevel":[0.8,0.9],
                  "reg_alpha":[0,0.1],
                  "max_delta_step":[0.1],
                  "seed":[0]}

    min_score = 100
    min_params = None

    
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info("params:{}".format(params))
        
        logloss_score_list = []
        gini_score_list = []
        for train_idx, valid_idx in cv_model.split(x_train,y_train):
            
        
            trn_x = x_train.iloc[train_idx,:]
            val_x = x_train.iloc[valid_idx,:]

            #yはすでにNumpy
            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]


            clf = xgb.sklearn.XGBClassifier(**params)
            clf.fit(trn_x,
                    trn_y,
                    eval_set =[(val_x,val_y)],
                    early_stopping_rounds=100,
                    eval_metric = gini_xgb)

            pred = clf.predict_proba(val_x)[:,1]

            sc_logloss = log_loss(val_y, pred)
            sc_gini = - gini(val_y, pred)

            logloss_score_list.append(sc_logloss)
            gini_score_list.append(sc_gini)

            logger.debug("logloss:{},gini:{}".format(sc_logloss, sc_gini))
            break

        logloss_score = np.mean(logloss_score_list)
        gini_score = np.mean(gini_score_list)
        if min_score > gini_score:
            min_score = gini_score
            min_params = params
        
        logger.info("logloss:{},gini:{}".format(logloss_score, gini_score))
        logger.info("current min score:{}, min_params:{} ".format(min_score, min_params))
        
    logger.info("minimum params :{}".format(min_params))   
    logger.info("minimum gini score :{}".format(min_score))   

    clf = xgb.sklearn.XGBClassifier(**min_params)
    clf.fit(x_train, y_train)

    logger.info("train end")


    df = load_test_data()

    x_test = df[use_cols].sort_values("id")

    logger.info("test data load end {}".format(x_test.shape))
    pred_test = clf.predict_proba(x_test)[:,1]

    df_submit = pd.read_csv(SAMPLE_SABMIT_FILE).sort_values("id")
    logger.info("submit sanple data load end {}".format(df_submit.shape))
    df_submit["target"] = pred_test

    df_submit.to_csv(DIR + "submit.csv", index=False)
    logger.info("end")
