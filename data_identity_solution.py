import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train_HK6lq50.csv')
test = pd.read_csv('test_2nAIblo.csv')

train['trainee_engagement_rating'] = train.groupby('trainee_id')['trainee_engagement_rating'].apply(lambda x: x.fillna(x.mean()))
train['trainee_engagement_rating'] = train['trainee_engagement_rating'].fillna(train['trainee_engagement_rating'].mean())

test['trainee_engagement_rating'] = test.groupby('trainee_id')['trainee_engagement_rating'].apply(lambda x: x.fillna(x.mean()))
test['trainee_engagement_rating'] = test['trainee_engagement_rating'].fillna(test['trainee_engagement_rating'].mean())

train['trainee_engagement_rating'] = train['trainee_engagement_rating'].astype(int)
test['trainee_engagement_rating'] = test['trainee_engagement_rating'].astype(int)

train['program_id'] = [x[-1] for x in train.program_id ]
test['program_id'] = [x[-1] for x in test.program_id ]

train['age'] = train['age'].fillna(train['age'].mean())
test['age'] = test['age'].fillna(train['age'].mean())
#train['age'] = train['age'].replace(to_replace = np.nan, value = -1)
#test['age'] = test['age'].replace(to_replace = np.nan, value = -1) 
train['age'] = train['age'].astype(int)
test['age'] = test['age'].astype(int)

train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

X = train
y = train.is_pass
X = train.drop(['is_pass'],axis = 1)

from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
categorical_features_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

model=CatBoostClassifier(iterations=1020, depth=6, learning_rate=0.06, loss_function= 'Logloss')
model.fit(X, y,cat_features=categorical_features_indices,plot=True)
#model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test,y_test),plot=True)
#pred=model.predict_proba(X_test)[:,1]
#score = roc_auc_score(y_test,pred)
#print('roc_auc_score',score)

#predict probabilities of is_pass
pred=model.predict_proba(test)[:,1]

#create a csv submission file
submission=pd.read_csv("sample_submission_vaSxamm.csv")
submission['is_pass']=pred
submission.to_csv('agemean_cb_cf14_610206+.csv', index=False)
