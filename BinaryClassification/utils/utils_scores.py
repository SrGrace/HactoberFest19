from sklearn.metrics import r2_score, roc_auc_score, classification_report
import pandas as pd


def scores(y_true, y_predict):
    r2 = r2_score(y_true, y_predict)
    auc = roc_auc_score(y_true, y_predict)

    y_true = y_true.reset_index(drop=True)
    test_result = pd.concat([y_true, pd.DataFrame(y_predict)], axis=1)
    classreport = classification_report(y_true=y_true, y_pred=y_predict)

    print("R2 score is {}".format(r2))
    print("auc score is {}".format(auc))
    print("result mapping true value to predicted value \n {}\n".format(test_result))
    print("Classification Report \n {}\n ".format(classreport))
