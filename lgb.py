import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,accuracy_score

import numpy as np

kf = KFold(n_splits=5,shuffle=True)
x=np.load('data_concat/data_concat.npy')
y=np.load('data_concat/label_concat.npy')

params={
    'learning_rate':0.05,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':10,
    'num_leaves':128,
    'objective':'multiclass',
    'num_class':9,
    'num_threads':8,
    'is_unbalance':True
}

for train_index,test_index in kf.split(x):
    X_train=x[train_index]
    y_train=y[train_index]
    X_test=x[test_index]
    y_test=y[test_index]

    train_data=lgb.Dataset(X_train,label=y_train)
    validation_data=lgb.Dataset(X_test,label=y_test)
    clf=lgb.train(params,train_data,1000,valid_sets=[validation_data])

    y_pred=clf.predict(X_test)

    y_argmax=y_pred.argmax(axis=1)
    select_index=(y_argmax==y_test)

    y_argmax_select=y_argmax[np.where(select_index)]
    acc_mean=0.

    for i in range(9):
        acc_item=(y_argmax_select==i).sum()*1.0/(y_test==i).sum()
        acc_mean+=acc_item
    
    acc=acc_mean/9

    clf.save_model('models/model_%.4f.txt'%(acc))
