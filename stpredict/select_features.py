import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import re

warnings.filterwarnings("once")

def select_features(data, ordered_covariates_or_features):
    if isinstance(data, str):    # if the input named 'data' is a string (is a directory address)
        data = pd.read_csv(data)
    
    output_data = pd.DataFrame()    # final dataframe to be returned

    # selecting temporal and futuristic features or covariates from the ordered_covariates_or_features list
    check_list = [item for item in ordered_covariates_or_features if item.count(' ') != 0]

    # type_flag for detecting feature type (False) or covariate type (True)
    # check if all elements in check_list meet the condition for being covariate type
    type_flag = all(re.search(' t$', element) or re.search(' t[+]$', element) for element in check_list)
    
    if type_flag == False:    # ordered_features
        output_data = data[ordered_covariates_or_features]

    elif type_flag == 1:    # ordered_covariates
        for covariate_name in ordered_covariates_or_features:
            # making a dataframe (tmp_df) containing the desirable columns
            if not(covariate_name in check_list):
                tmp_df = data[[covariate_name]]
            else:
                tmp_df = data.filter(regex=covariate_name)
            output_data = pd.concat([output_data, tmp_df], axis=1)    # concat two dataframes
    
    output_data = pd.concat([data[['spatial id', 'temporal id', 'Target', 'Normal target']], output_data], axis=1)

    return output_data
