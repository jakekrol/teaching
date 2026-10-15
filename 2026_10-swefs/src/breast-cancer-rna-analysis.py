#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
parser=argparse.ArgumentParser()
parser.add_argument("--input", default="../data/METABRIC_RNA_Mutation.csv",
                    help="Kaggle breast cancer dataset")
parser.add_argument("--output", default="../img/breast-cancer-rna-analysis.png")
parser.add_argument("--seed", default=0)
args = parser.parse_args()
args.seed = int(args.seed) # cast str to int when not default

# hypothesis: gene expression can separate ER+/- breast cancers
# note: surprisingly, ESR1 (estrogen receptor gene) expression is not in dataset. 
# approach: i) count num. ER+/- cases, ii) fit and evaluate logistic regression model using test auroc
# figures: i) bar plot ER+/-, ii) model test auroc, and iii) top model coef bar

def main():
    np.random.seed(args.seed)
    df = pd.read_csv(args.input) # read data into pandas DataFrame
    ### organizing columns into 3 categories: i) metadata, ii) gene expression, and
    # iii) gene mutation
    mutation_col_suffix = '_mut'
    col_idx_gene_expr_start = -1 # first column of gene expression
    col_idx_mut_start = -1 # first column of gene mutation
    col_idx_er_status = -1 # target
    for i,col in enumerate(df.columns):
        if col == 'er_status':
            col_idx_er_status = i
        if col == 'brca1':
            col_idx_gene_expr_start = i
        if col.endswith(mutation_col_suffix):
            col_idx_mut_start = i
            break # exit loop
    # verify col indices were discovered
    assert (
        all([
            col_idx_gene_expr_start != -1,
            col_idx_mut_start != -1,
            col_idx_er_status != -1
        ])
    )
    ### target class balance
    target = df.iloc[:,col_idx_er_status]
    target_counts = target.value_counts() # class balance
    assert (target.isna().sum() == 0)
    # encode class labels
    def label2int(label):
        d = {'Positive': 1, 'Negative': 0}
        return d[label]
    y = target.map(label2int).values
    ### stratified k-fold cross validation
    # stratify approximately preserves class balance in each fold
    x = df.iloc[:,col_idx_gene_expr_start:col_idx_mut_start] # 1904 x 489
    assert (x.isna().sum().sum() == 0) # bool matrix > column NA sums > total NA values
    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=5, # number of train test split pairs
        random_state=args.seed,
        shuffle=True # relevant here since samples are sorted by cohort
    )
    # best practice is to aggregate results across folds, then evaluate
    y_pred_base,y_pred_ridge = [],[] # test predictions
    y_true= [] # test truth 
    # for n_splits iterations, fit on train and eval on test 
    for i, (idx_train, idx_test) in enumerate(skf.split(x,y)):
        # split data
        x_train = x.iloc[idx_train, :] # x is still DataFrame
        x_test = x.iloc[idx_test, :]
        y_train = y[idx_train] # y is numpy array
        y_test = y[idx_test]
        # models
        model_ridge = sklearn.linear_model.LogisticRegression(
            l1_ratio=0, # ridge regression (penalize L2 norm of weights); use ratio because of elastic net compatibility
            C=1, # no hyperparameter tuning here, just use default penalty coef
            random_state=args.seed
        )
        model_base = sklearn.linear_model.LogisticRegression(
            random_state=args.seed,
            C=np.inf # no penalty term in objective (1/\infty coef)
        ) 
        # train
        model_ridge.fit(x_train,y_train)
        model_base.fit(x_train,y_train)
        # predict
        y_pred_ridge.extend(model_ridge.predict_proba(x_test)[:,1]) # returns n by 2 for binary classification task
        y_pred_base.extend(model_base.predict_proba(x_test)[:,1])
        y_true.extend(y_test)
    assert (len(y_true) == len(y_pred_ridge) == len(y_pred_base)) # sanity check: prediction and truth are equal in size
    # ROC analysis
    base_fpr, base_tpr, base_thresh = sklearn.metrics.roc_curve(y_true, y_pred_base, pos_label=1)
    base_auroc = round(sklearn.metrics.roc_auc_score(y_true, y_pred_base),2)
    ridge_fpr, ridge_tpr, ridge_thresh = sklearn.metrics.roc_curve(y_true, y_pred_ridge, pos_label=1)
    ridge_auroc = round(sklearn.metrics.roc_auc_score(y_true, y_pred_ridge),2)
    # top coefficients (ridge)
    # here, i will just fit on the whole dataset and inspect coef, ...
    # in practice, you would probably want to dig into coefs during the k-fold stratified cv step
    model_ridge_all = sklearn.linear_model.LogisticRegression(
            l1_ratio=0,
            C=1, 
            random_state=args.seed
        )
    model_ridge_all.fit(x,y)
    coefs = model_ridge_all.coef_.ravel()
    series = pd.Series(coefs,index=x.columns.tolist())
    series = series.sort_values(ascending=False)
    ### visualization
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
    ax=ax.flatten() # collapses from 2d array into 1d array, makes indexing subplots simpler
    # ER +/- bar plot
    ax[0].bar(x=target_counts.index,height=target_counts.values,color='black')
    ax[0].set_title("ER cases",loc="left")
    ax[0].set_ylabel('Count')
    ax[1].plot(base_fpr,base_tpr,color='red',label='Base')
    ax[1].plot(ridge_fpr,ridge_tpr,color='blue',label='Ridge')
    ax[1].plot([0,1], [0,1], label='Baseline',color='black',linestyle='--')
    ax[1].set_xlabel('False positive rate')
    ax[1].set_ylabel('True positive rate')
    ax[1].set_title("Model separation of ER+/-",loc="left")
    ax[1].text(0.2, 0.7, f"auROC base: {base_auroc}\nauROC ridge: {ridge_auroc}",
               transform=ax[1].transAxes)
    ax[2].bar(x=series.head(5).index,height=series.head(5).values,color='black')
    ax[2].set_ylabel("Log. reg. coefficient")
    ax[2].set_xlabel("Gene")
    for i in range(len(ax)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(args.output)

if __name__ == "__main__":
    main()
