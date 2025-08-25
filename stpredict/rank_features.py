import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import sys

    import pandas as pd
    import numpy as np

    from .configurations import *
    from .get_target_quantities import get_target_quantities


def rank_features(data,
                  ranking_method: str = 'mRMR',
                  forced_covariates: list = []):

    # inputs checking
    # data checking
    if not (isinstance(data, pd.DataFrame) or isinstance(data, str)):
        sys.exit("Error: The input 'data' must be a dataframe or an address to a dataframe.")
    # ranking_method checking
    if ranking_method not in RANKING_METHODS:
        sys.exit(f"Error: The input 'ranking_method' is not valid. Valid options are {RANKING_METHODS}.")
    # forced_covariates checking
    if not isinstance(forced_covariates, list):
        sys.exit("Error: The input 'forced_covariates' must be a list of covariates or an empty list.")

    if isinstance(data, str):
        try:
            data = pd.read_csv(data)
        except Exception as e:
            sys.exit(str(e))
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        sys.exit("The data input format is not valid.")

    # forced_covariates manipulation
    forced_covariates = list(set(forced_covariates))
    forced_covariates = [forced_covariate
                         for forced_covariate in forced_covariates
                         if forced_covariate is not None and forced_covariate != '']

    if TARGET_COLUMN_NAME not in data.columns.values:
        _, _, _, data = get_target_quantities(data.copy())

    data.drop(NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
    data.drop(NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)

    forced_features = []
    for forced_covariate in forced_covariates:
        if forced_covariate in data.columns.values:
            forced_features.append(forced_covariate)
        if forced_covariate + ' t' in data.columns.values:
            forced_features.append(forced_covariate + ' t')
        for column_name in data.columns.values:
            if column_name.startswith(forced_covariate + ' t-'):
                forced_features.append(column_name)
        for column_name in data.columns.values:
            if column_name.startswith(forced_covariate + ' t+'):
                forced_features.append(column_name)
    forced_features.sort()

    data.drop(forced_features, axis=1, inplace=True)

    if ranking_method == 'mRMR':
        cor = data.corr().abs()
        valid_feature = cor.index.drop([TARGET_COLUMN_NAME])
        overall_rank_df = pd.DataFrame(index=cor.index, columns=['mrmr_rank'])
        for i in cor.index:
            overall_rank_df.loc[i, 'mrmr_rank'] = \
                cor.loc[i, TARGET_COLUMN_NAME] - cor.loc[i, valid_feature].mean()
        overall_rank_df = overall_rank_df.sort_values(by='mrmr_rank', ascending=False)
        overall_rank = overall_rank_df.index.tolist()
        final_rank = overall_rank[0:2]
        overall_rank = overall_rank[2:]
        while len(overall_rank) > 0:
            temp = pd.DataFrame(index=overall_rank, columns=['mrmr_rank'])
            for i in overall_rank:
                temp.loc[i, 'mrmr_rank'] = cor.loc[i, TARGET_COLUMN_NAME] - cor.loc[i, final_rank[1:]].mean()
            temp = temp.sort_values(by='mrmr_rank', ascending=False)
            final_rank.append(temp.index[0])
            overall_rank.remove(temp.index[0])
        final_rank.remove(TARGET_COLUMN_NAME)
        ix = final_rank
    elif ranking_method == 'correlation':
        ix = data.corr().abs().sort_values(TARGET_COLUMN_NAME, ascending=False).index.drop(
            [TARGET_COLUMN_NAME])
    else:
        target = np.copy(data[TARGET_COLUMN_NAME].to_numpy())
        variance_rank = dict()
        for column in data.columns.values:
            if column == TARGET_COLUMN_NAME:
                continue
            feature_vec = np.copy(data[column].to_numpy())
            min_v, max_v, n_sec = np.min(feature_vec), np.max(feature_vec), 10
            if max_v > min_v:
                step = (max_v - min_v) / n_sec
                feature_vec = np.floor((feature_vec - min_v) / step) * step + min_v
            cnd_ex = []
            cnd_var = []
            for ix in range(len(target)):
                idx = feature_vec == feature_vec[ix]
                cnd_ex.append(np.mean(target[idx]))
                cnd_var.append(np.var(target[idx]))
            cnd_expectation, cnd_variance = np.array(cnd_ex), np.array(cnd_var)
            variance_method_value = np.var(cnd_expectation) / np.var(cnd_variance) if np.var(cnd_variance) > 0 else 1
            if variance_method_value < 0:
                variance_method_value = 0
            if variance_method_value > 1:
                variance_method_value = 1
            variance_rank[column] = variance_method_value
        variance_rank = {k: v for k, v in sorted(variance_rank.items(), key=lambda item: -item[1])}
        ix = variance_rank.keys()
    ranked_features = forced_features
    ranked_features.extend(ix)
    return ranked_features
