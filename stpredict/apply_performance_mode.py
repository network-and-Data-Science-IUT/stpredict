import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np

def apply_performance_mode(training_target, test_target, training_prediction, test_prediction,
                           performance_mode, same_train_test = False):
    
    '''
    ::: input :::
    training_target : a data frame including columns 'spatial id', 'temporal id', 'Normal target'
                        from the training set (extra columns allowed)
                        
    test_target : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the test set (extra columns allowed)
                        
    training_prediction : list of predicted values for the training set
    test_prediction : list of predicted values for the test set
    
    performance_mode : 'normal' , 'cumulative' , 'moving_average + x' the desired mode of the target variable
                        when calculating the performance
                        
    same_train_test : a flag to identify the state that the same data set is used for training and test
                        
    ::: output :::
    
    training_target : a data frame with the same columns as input training_target with the modified
                        values for the 'Normal target' based on performance_mode
                        
    test_target : a data frame with the same columns as input test_target with the modified
                        values for the 'Normal target' based on performance_mode
                        
    training_prediction : list of modified predicted values for the training set based on performance_mode
    test_prediction : list of modified predicted values for the test set based on performance_mode
    
    '''
    column_names_list = list(training_target.columns)
    
    if performance_mode == 'normal':
        return training_target, test_target, training_prediction, test_prediction
        
    # decode performance mode to get the window size of the moving average
    if performance_mode.startswith('moving_average'):
        if len(performance_mode.split(' + ')) > 1:
            window = performance_mode.split(' + ')[-1]
            performance_mode = 'moving_average'
        else:
            raise ValueError("For the moving average performance_mode the window size must also be specifid in the performance_mode with the format 'moving_average + x' where x is the window size.")
        try:
            window = int(window)
        except ValueError:
            raise ValueError("The specified window for the moving average performance_mode is not valid.")
    
    training_target.loc[:,('type')] = 1
    test_target.loc[:,('type')] = 2
    
    training_target.loc[:,('prediction')] = training_prediction
    test_target.loc[:,('prediction')] = test_prediction
    
    data = training_target.append(test_target)
    training_temporal_ids = training_target['temporal id'].unique()
    test_temporal_ids = test_target['temporal id'].unique()
    
    if same_train_test == True:
        data = training_target.copy()
        test_temporal_ids = []
        test_prediction = []
    
    data = data.sort_values(by = ['temporal id','spatial id'])
    temporal_ids = data['temporal id'].unique()
    
    # if performance mode is cumulative, the cumulative values of the target and prediction is calculated
    if performance_mode == 'cumulative':
        
        # next 6 lines modify the target real values mode to cumulative mode
        cumulative_target_df = data.copy().pivot(index='temporal id', columns='spatial id', values='Normal target')
        cumulative_target_df = cumulative_target_df.cumsum()
        cumulative_target_df = pd.melt(cumulative_target_df.reset_index(), id_vars='temporal id', value_vars=list(cumulative_target_df.columns),
                                         var_name='spatial id', value_name='Normal target')
        data = data.drop(['Normal target'], axis = 1)
        data = pd.merge(data, cumulative_target_df, how = 'left')
        
        # practical accessible values (target real values in training set and predicted values in test set) will be used 
        # to get the cumulative target (predicted)
        data['train_real_test_prediction'] = data['Normal target']
        data.loc[data['type'] == 2,'train_real_test_prediction'] = data.loc[data['type'] == 2,'prediction']
        data = data.sort_values(by = ['temporal id','spatial id'])
        
        dates = data['temporal id'].unique()
        for index in range(len(dates)):
            date = dates[index + 1]
            past_date = dates[index]
            data.loc[
                data['temporal id'] == date, 'prediction'] = list(np.array(
                data.loc[data['temporal id'] == date, 'prediction']) + np.array(
                data.loc[data['temporal id'] == past_date, 'train_real_test_prediction']))
            if date in test_temporal_ids:
                data.loc[data['temporal id'] == date, 'train_real_test_prediction'] = list(data.loc[data['temporal id'] == date, 'prediction'])

            if index == len(dates) - 2:
                break
        
    elif performance_mode == 'moving_average':
        if window > len(temporal_ids):
            raise ValueError("The specified window for the moving average performance_mode is too large for the input data.")
        
        number_of_spatial_units = len(data['spatial id'].unique())
        
        # practical accessible values (target real values in training set and predicted values in test set) will be used 
        # to get the moving average target (predicted)
        data['train_real_test_prediction'] = data['Normal target']
        data.loc[data['type'] == 2,'train_real_test_prediction'] = data.loc[data['type'] == 2,'prediction']
        data = data.sort_values(by = ['temporal id','spatial id'])

        dates = data['temporal id'].unique()
        for index in range(len(dates)):
            ind = index + window - 1
            date = dates[ind]
            
            for i in range(window - 1):
                past_date = dates[ind - (i + 1)]
                data.loc[
                    data['temporal id'] == date, 'prediction'] = list(np.array(
                    data.loc[data['temporal id'] == date, 'prediction']) + np.array(
                    data.loc[data['temporal id'] == past_date, 'train_real_test_prediction']))
                
            data.loc[data['temporal id'] == date, 'prediction'] = list(np.array(
                data.loc[data['temporal id'] == date, 'prediction'])/window)
            
            if ind == len(dates) - 1:
                break
        
        # next 6 lines modify the target real values mode to moving average mode
        moving_avg_target_df = data.copy().pivot(index='temporal id', columns='spatial id', values='Normal target')
        moving_avg_target_df = moving_avg_target_df.rolling(window).mean()
        moving_avg_target_df = pd.melt(moving_avg_target_df.reset_index(), id_vars='temporal id', value_vars=list(moving_avg_target_df.columns),
                                         var_name='spatial id', value_name='Normal target')
        data = data.drop(['Normal target'], axis = 1)
        data = pd.merge(data, moving_avg_target_df, how = 'left')
        
        data = data.sort_values(by = ['temporal id', 'spatial id'])
        # data = data.iloc[(window-1)*number_of_spatial_units:] # ?????????????????????? #
        
    else:
        raise ValueError("Specified performance_mode is not valid.")
        
    data = data.sort_values(by=['temporal id', 'spatial id'])
    training_set = data[data['type'] == 1]
    test_set = data[data['type'] == 2]
    
    if same_train_test == True:
        test_set = training_set.copy()
    
    if (len(test_set) < 1) and (performance_mode == 'moving_average'):
        raise Exception("The number of remaining instances in the test set is less than one when applying moving average performance_mode (the first 'window - 1' temporal units is removed in the process)")
    if (len(training_set) < 1) and (performance_mode == 'moving_average'):
        raise Exception("The number of remaining instances in the training set is less than one when applying moving average performance_mode (the first 'window - 1' temporal units is removed in the process).")

    training_prediction = list(training_set['prediction'])
    test_prediction = list(test_set['prediction'])

    training_target = training_set.drop(['type','prediction','train_real_test_prediction'], axis = 1)
    test_target = test_set.drop(['type','prediction','train_real_test_prediction'], axis = 1)
    
    training_target = training_target[column_names_list]
    test_target = test_target[column_names_list]
    
    return training_target, test_target, training_prediction, test_prediction

