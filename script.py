import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x = []
for col in train.columns:
    if train[col].std() == 0:
        x.append(col)

train.drop(x, axis=1, inplace=True)
test.drop(x, axis=1, inplace=True)

x = []
cols = train.columns
for i in range(len(cols)-1):
    vals = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(vals, train[cols[j]].values):
            x.append(cols[j])

train.drop(x, axis=1, inplace=True)
test.drop(x, axis=1, inplace=True)

y_train = train['TARGET'].values
X_train = train.drop(['ID','TARGET'], axis=1).values

y_test = test['ID']
X_test = test.drop(['ID'], axis=1).values

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=600,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.6815,
 colsample_bytree=0.701,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgtrain = xgb.DMatrix(X_train, label=y_train)
cvresult = xgb.cv(xgb1.get_xgb_params(), xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
metrics=['auc'], early_stopping_rounds=50, show_progress=False)
xgb1.set_params(n_estimators=cvresult.shape[0])
xgb1.fit(X_train, y_train, eval_metric='auc')
output = xgb1.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":y_test, "TARGET":output})
submission.to_csv("submission.csv", index=False)