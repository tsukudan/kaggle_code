import pandas as pd
import numpy as np
from logging import StreamHandler, Formatter ,FileHandler, getLogger,DEBUG, INFO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = "result_tmp/"
SAMPLE_SABMIT_FILE = "../input/sample_submission.csv"



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

    clf = LogisticRegression(random_state=0)
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

    
