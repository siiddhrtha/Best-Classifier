#!/usr/bin/env python
# coding: utf-8

# In[121]:


pip install lightgbm


# In[75]:


pip install catboost


# In[76]:


import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.simplefilter('ignore')

from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from mlxtend.classifier import StackingCVClassifier


# In[77]:




RANDOM_SEED = 2021
PROBAS = True
FOLDS = 5
N_ESTIMATORS = 1000

TARGET = 'class'


# In[120]:



train = pd.read_csv(r"C:\Users\siddh\Downloads\Android Malware Dataset\drebin-215-dataset-5560malware-9476-benign.csv")
#test = pd.read_csv('/content/drive/MyDrive/Datasets/drebin-215-dataset-5560malware-9476-benign.csv')


# In[119]:


train['class'].value_counts()


# In[89]:


X = train.drop(['class'], axis=1)
y = train['class']


print (f'X:{X.shape} y: {y.shape} \n')

X_train, X_test, y_train, y_test = train_test_split(train[train.columns[:len(train.columns)-1]].to_numpy(),train[train.columns[-1]].to_numpy(),test_size = 0.2, random_state = RANDOM_SEED)
print (f'X_train:{X_train.shape} y_train: {y_train.shape}')
print (f'X_test:{X_test.shape} y_test: {y_test.shape}')


# In[90]:


classes,count = np.unique(train['class'],return_counts=True)
#Perform Label Encoding
lbl_enc = LabelEncoder()
print(lbl_enc.fit_transform(classes),classes)
train = train.replace(classes,lbl_enc.fit_transform(classes))

#Dataset contains special characters like ''?' and 'S'. Set them to NaN and use dropna() to remove them
train=train.replace('[?,S]',np.NaN,regex=True)
print("Total missing values : ",sum(list(train.isna().sum())))
train.dropna(inplace=True)
for c in train.columns:
    train[c] = pd.to_numeric(train[c])
train


# In[91]:


print("Total Features : ",len(train.columns)-1)


# In[92]:


plt.bar(classes,count)
plt.title("Class balance")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()


# In[93]:


lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': 10000,
    'objective': 'binary',
    'learning_rate': 0.02,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
    'device': 'gpu'
}

cb_params = {
    'max_depth':6,
    'max_ctr_complexity': 5,
    'num_trees': 50000,
    'od_wait': 500,
    'od_type':'Iter', 
    'learning_rate': 0.04,
    'min_data_in_leaf': 3,
    'task_type': 'GPU'
}


rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED
}


# In[94]:


cl3 = RandomForestClassifier(**rf_params)
cl4 = DecisionTreeClassifier(max_depth = 5)
cl5 = CatBoostClassifier(task_type = 'CPU', verbose = None, logging_level = 'Silent')
cl6 = LGBMClassifier()
cl7 = ExtraTreesClassifier(bootstrap=False, criterion='entropy', max_features=0.55, min_samples_leaf=8, min_samples_split=4, n_estimators=100) # Optimized using TPOT
cl8 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = RANDOM_SEED)


# In[95]:


classifiers = {
    "RF": cl3,
    "DecisionTree": cl4,
    "CatBoost": cl5,
    "LGBM": cl6,
    "ExtraTrees": cl7,
    "MLP": cl8
}


# In[96]:


mlr = LogisticRegression()


# In[97]:



models_scores_results, models_names = list(), list() 


# In[98]:


print(">>>> Training started <<<<")
for key in classifiers:
    classifier = classifiers[key]
    scores = model_selection.cross_val_score(classifier, X_train, y_train, cv = FOLDS, scoring='accuracy')
    models_scores_results.append(scores)
    models_names.append(key)
    print("[%s] - accuracy: %0.5f " % (key, scores.mean()))
    classifier.fit(X_train, y_train)
    
    # Save classifier for prediction 
    classifiers[key] = classifier
    


# In[99]:


plt.figure(figsize=(7, 5))
plt.boxplot(models_scores_results, labels=models_names, showmeans=True)
plt.show()


# In[100]:


taken_classifiers = ["RF", "DecisionTree","CatBoost","LGBM", "ExtraTrees","MLP"]


# In[101]:


#This function searches best stacking configuration
def best_stacking_search():
   cls_list = []
   best_auc = -1
   i=0

   best_cls_experiment = list()

   print(">>>> Training started <<<<")

   for cls_comb in range(2, len(taken_classifiers)+1):
       for subset in itertools.combinations(taken_classifiers, cls_comb):
           cls_list.append(subset)

   print(f"Total number of model combination: {len(cls_list)}")


   for cls_exp in cls_list:
       cls_labels = list(cls_exp)

       classifier_exp = []
       for ii in range(len(cls_labels)):
           label = taken_classifiers[ii]
           classifier = classifiers[label]
           classifier_exp.append(classifier)


       sclf = StackingCVClassifier(classifiers = classifier_exp,
                                   shuffle = False,
                                   use_probas = True,
                                   cv = FOLDS,
                                   meta_classifier = mlr,
                                   n_jobs = -1)

       scores = model_selection.cross_val_score(sclf, X_train, y_train, cv = FOLDS, scoring='accuracy')

       if scores.mean() > best_auc:
           best_cls_experiment = list(cls_exp)
       i += 1
       print(f"  {i} - Stacked combination - Acc {cls_exp}: {scores.mean():.5f}")
       
   return best_cls_experiment


# In[102]:


get_ipython().run_cell_magic('time', '', '\nbest_cls_experiment = ["RF", "DecisionTree","CatBoost","LGBM", "ExtraTrees","MLP"]')


# In[103]:


print(f'The best models configuration: {best_cls_experiment}')

classifier_exp = []
for label in best_cls_experiment:
        classifier = classifiers[label]
        classifier_exp.append(classifier)


# In[104]:


classifier_exp


# In[105]:



scl = StackingCVClassifier(classifiers= classifier_exp,
                            meta_classifier = mlr, # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)

scores = model_selection.cross_val_score(scl, X_train, y_train, cv = FOLDS, scoring='accuracy')
models_scores_results.append(scores)
models_names.append('scl')
print("Meta model (slc) - accuracy: %0.5f " % (scores.mean()))
scl.fit(X_train, y_train)

top_meta_model = scl
base_acc = scores.mean()


# In[106]:


scl.get_params().keys()


# In[107]:


def meta_best_params_search():

    scl_params = {'meta_classifier__C': [0.001, 0.01, 0.1, 1, 10]}

    print(">>>> Searching for best parameters started <<<<")

    grid = GridSearchCV(estimator=scl, 
                        param_grid= scl_params, 
                        cv=5,
                        refit=True)
    grid.fit(X_train, y_train)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r" % (grid.cv_results_[cv_keys[0]][r], grid.cv_results_[cv_keys[1]][r] / 2.0, grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.5f' % grid.best_score_)
    return grid, grid.best_score_


# In[122]:


hyper_meta_model, hyper_acc = meta_best_params_search()


# In[108]:


scl = StackingCVClassifier(classifiers= classifier_exp,
                            meta_classifier = LogisticRegression(C = 0.1), # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)

scores = model_selection.cross_val_score(scl, X_train, y_train, cv = FOLDS, scoring='accuracy')
print("Meta model (slc) - accuracy: %0.5f " % (scores.mean()))
scl.fit(X_train, y_train)
top_meta_model = scl
# SCENARIO 2.

classifiers["scl"] = top_meta_model


# In[109]:


preds = pd.DataFrame()

for key in classifiers:
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    preds[f"{key}"] = y_pred
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"{key} -> AUC: {auc:.5f}")

preds[TARGET] = pd.DataFrame(y_test).reset_index(drop=True)


# In[110]:



def plot_model_AUC(cls_models):
    NUM_CLASS = len(cls_models)
    sns.set(font_scale = 1)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                   "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                   'ytick.color': '0.4'})

    f, ax = plt.subplots(figsize=(20, 5), nrows=1, ncols = NUM_CLASS)

    for key, counter in zip(cls_models, range(NUM_CLASS)):

        y_pred = preds[key]

        auc = metrics.roc_auc_score(y_test, y_pred)
        textstr = f"AUC: {auc:.3f}"


        false_pred = preds[preds[TARGET] == 0]
        sns.distplot(false_pred[key], hist=True, kde=True, 
                     bins=int(50), color = 'red', 
                     hist_kws={'edgecolor':'black'}, ax = ax[counter])


        true_pred = preds[preds[TARGET] == 1]
        sns.distplot(true_pred[key], hist=True, kde=True, 
                     bins=int(50), color = 'green', 
                     hist_kws={'edgecolor':'black'}, ax = ax[counter])


        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                        verticalalignment = "top", bbox=props)

        ax[counter].set_title(f"{key}")
        ax[counter].set_xlim(0,1)
        ax[counter].set_xlabel("probability")

    plt.tight_layout()


# In[111]:


plot_model_AUC(classifiers)


# In[112]:


r_probs = [0 for _ in range(len(y_test))]
rf_probs = cl3.predict_proba(X_test)
dt_probs = cl4.predict_proba(X_test)
cb_probs = cl5.predict_proba(X_test)
lgbm_probs = cl6.predict_proba(X_test)
et_probs = cl7.predict_proba(X_test)
mlp_probs = cl8.predict_proba(X_test)
scl_probs = scl.predict_proba(X_test)


# In[113]:


rf_probs = rf_probs[:, 1]
dt_probs = dt_probs[:, 1]
cb_probs = cb_probs[:, 1]
lgbm_probs = lgbm_probs[:, 1]
et_probs = et_probs[:, 1]
mlp_probs = mlp_probs[:, 1]
scl_probs = scl_probs[:, 1]


# In[114]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[115]:


r_auc = roc_auc_score(y_test, r_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
dt_auc = roc_auc_score(y_test, dt_probs)
cb_auc = roc_auc_score(y_test, cb_probs)
lgbm_auc = roc_auc_score(y_test, lgbm_probs)
et_auc = roc_auc_score(y_test, et_probs)
mlp_auc = roc_auc_score(y_test, mlp_probs)
scl_auc = roc_auc_score(y_test, scl_probs)


# In[116]:


r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
cb_fpr, cb_tpr, _ = roc_curve(y_test, cb_probs)
lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_probs)
et_fpr, et_tpr, _ = roc_curve(y_test, et_probs)
mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_probs)
scl_fpr, scl_tpr, _ = roc_curve(y_test, scl_probs)


# In[117]:


plt.figure(figsize=(10, 7))
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.',linestyle='-', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(dt_fpr, dt_tpr, marker='o',linestyle=':', label='Decision Tree (AUROC = %0.3f)' % dt_auc)
plt.plot(cb_fpr, cb_tpr, marker='.',linestyle='-.', label='CatBoost (AUROC = %0.3f)' % cb_auc)
plt.plot(lgbm_fpr, lgbm_tpr, marker='.',linestyle='-.', label='Light Gradient Boosting Machine (AUROC = %0.3f)' % lgbm_auc)
plt.plot(et_fpr, et_tpr, marker='.',linestyle=':', label='ExtraTrees (AUROC = %0.3f)' % et_auc)
plt.plot(mlp_fpr, mlp_tpr, marker='.',linestyle=':', label='Multilayer perceptron(AUROC = %0.3f)' % mlp_auc)
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Show legend
plt.legend() # 


# Show plot
plt.show()


# In[118]:


plt.figure(figsize=(7, 5))
plt.plot(r_fpr, r_tpr, linestyle='--',color='c')
plt.plot(dt_fpr, dt_tpr, marker='v',linestyle=':',color='g' ,label='Least predicted Model (AUROC = %0.3f)' % dt_auc)
plt.plot(scl_fpr, scl_tpr, marker='.',linestyle='-',color='r', label=' Meta Model Classifier(AUROC = %0.3f)' % scl_auc)
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Show legend
plt.legend() # 


# Show plot
plt.show()


# In[ ]:




