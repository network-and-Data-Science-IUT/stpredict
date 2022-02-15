import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np

def get_trivial_values(train_true_values_df, validation_true_values_df, train_prediction,
                       validation_prediction, forecast_horizon, granularity, same_train_validation = False):
    
    '''
    :::inputs:::
    
    train_true_values_df : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the training set
    validation_true_values_df : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the test set           
    train_prediction : list of predicted values for the training set
    validation_prediction : list of predicted values for the test set
    forecast_horizon : number of temporal units in the future to be forecasted
    granularity : number of smaller scale temporal units which is averaged to get the values 
                    of bigger scale unit in the temporal transformation process
    same_train_validation : a flag to identify the state that the same data set is used for training and validation
    
    :::outputs:::
    
    train_true_values: a list target real values in the training set
    train_predicted_values: a list predicted values for the training set
    train_trivial_values: a list of trivial values for the training set
    validation_true_values: a list of target real values in the validation set
    validation_predicted_values: a list of predicted values for the validation set
    validation_trivial_values: a list of trivial values for the validation set
    
    '''
    
    train_true_values_df.loc[:,('prediction')] = train_prediction
    train_true_values_df.loc[:,('type')] = 1
    if same_train_validation == False:
        validation_true_values_df.loc[:,('prediction')] = validation_prediction
        validation_true_values_df.loc[:,('type')] = 2
        whole_data = train_true_values_df.append(validation_true_values_df)
    else:
        whole_data = train_true_values_df
    
    number_of_spatial_units = len(whole_data['spatial id'].unique())
    whole_data = whole_data.sort_values(by = ['temporal id', 'spatial id'])
    
    accessible_data = whole_data.copy().iloc[(forecast_horizon * granularity * number_of_spatial_units):,:]
    accessible_data['trivial values'] = list(whole_data.iloc[:-(forecast_horizon * granularity * number_of_spatial_units),:]['Normal target'])
    
    if same_train_validation == False:
        train_true_values = list(np.array(accessible_data.loc[accessible_data['type'] == 1,'Normal target']).reshape(-1))
        train_predicted_values = list(np.array(accessible_data.loc[accessible_data['type'] == 1,'prediction']).reshape(-1))
        train_trivial_values = list(np.array(accessible_data.loc[accessible_data['type'] == 1,'trivial values']).reshape(-1))

        validation_true_values = list(np.array(accessible_data.loc[accessible_data['type'] == 2,'Normal target']).reshape(-1))
        validation_predicted_values = list(np.array(accessible_data.loc[accessible_data['type'] == 2,'prediction']).reshape(-1))
        validation_trivial_values = list(np.array(accessible_data.loc[accessible_data['type'] == 2,'trivial values']).reshape(-1))
    else:
        train_true_values = list(np.array(accessible_data['Normal target']).reshape(-1))
        train_predicted_values = list(np.array(accessible_data['prediction']).reshape(-1))
        train_trivial_values = list(np.array(accessible_data['trivial values']).reshape(-1))

        validation_true_values = list(np.array(accessible_data['Normal target']).reshape(-1))
        validation_predicted_values = list(np.array(accessible_data['prediction']).reshape(-1))
        validation_trivial_values = list(np.array(accessible_data['trivial values']).reshape(-1))
        
    return train_true_values, train_predicted_values, train_trivial_values, validation_true_values, validation_predicted_values, validation_trivial_values

