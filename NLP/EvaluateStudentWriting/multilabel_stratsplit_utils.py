import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from collections import Counter

# This method uses the iterstrat library for multilabel stratification
def iterstrat_multilabel_stratified_kfold_cv_split(df_train, label_cols, num_folds, random_state):
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)    
    df_targets = df_train[label_cols]
    for fold, (train_index, val_index) in enumerate(mskf.split(df_train["id"], df_targets)):        
        df_train.loc[val_index, "kfold"] = fold
    return df_train

# This method uses the skmultilearn library for multilabel stratification
def skml_multilabel_stratified_kfold_cv_split(df_train, label_cols, num_folds):
    mskf = IterativeStratification(n_splits=num_folds, order=1)
    X = df_train["id"]
    y = df_train[label_cols]
    for fold, (train_index, val_index) in enumerate(mskf.split(X, y)):        
        df_train.loc[val_index, "kfold"] = fold
    return df_train

def get_train_val_split_stats(df, num_folds, label_cols):
    counts = {}
    for fold in range(num_folds):
        y_train = df[df.kfold != fold][label_cols].values
        y_val = df[df.kfold == fold][label_cols].values
        counts[(fold, "train_count")] = Counter(
                                        str(combination) for row in get_combination_wise_output_matrix(y_train, order=1) 
                                        for combination in row
                                    )
        counts[(fold, "val_count")] = Counter(
                                        str(combination) for row in get_combination_wise_output_matrix(y_val, order=1) 
                                        for combination in row
                                    )
    # View distributions
    df_counts = pd.DataFrame(counts).T.fillna(0)
    df_counts.index.set_names(["fold", "counts"], inplace=True)
    for fold in range(num_folds):
        train_counts = df_counts.loc[(fold, "train_count"), :]
        val_counts = df_counts.loc[(fold, "val_count"), :]
        val_train_ratio = pd.Series({i: val_counts[i] / train_counts[i] for i in train_counts.index}, name=(fold, "val_train_ratio"))
        df_counts = df_counts.append(val_train_ratio)
    df_counts = df_counts.sort_index() 
    return df_counts    