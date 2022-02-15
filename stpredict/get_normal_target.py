import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd 
    import numpy as np
    pd.options.mode.chained_assignment = None

def get_normal_target(training_target, test_target, training_prediction, test_prediction, target_mode,
                      target_granularity=None, same_train_test=False):
    """
    training_target : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the training set
    test_target : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the test set
    training_prediction : list of predicted values for the training set
    test_prediction : list of predicted values for the test set
    target_mode = 'normal' , 'cumulative' , 'moving average', 'differential' the mode of the target variable
    target_granularity : number of smaller temporal units which is averaged to get the moving average target
    same_train_test : a flag to identify the state that the same data set is used for training and test
    """

    if target_mode == 'normal':
        return training_target, test_target, training_prediction, test_prediction

    training_target.loc[:, ('type')] = 1
    test_target.loc[:, ('type')] = 2
    
    training_dates = list(training_target['temporal id'].unique())
    test_dates = list(test_target['temporal id'].unique())
    data = training_target.append(test_target)
    
    if same_train_test == True:
        test_dates = []
        test_prediction = []
        data = training_target
    
    # if target mode is cumulative we need to return the target variable to its original state
    if target_mode == 'cumulative':

        # practical accessible values (target real values in training set and predicted values in test set) will be used 
        # for returning the cumulative target (predicted) to original state
        data.loc[:, ('train_real_test_prediction')] = list(training_target['Target']) + list(test_prediction)
        data.loc[:, ('prediction')] = list(training_prediction) + list(test_prediction)
        
        data = data.sort_values(by=['temporal id', 'spatial id'])
        reverse_dates = data['temporal id'].unique()[::-1]

        for index in range(len(reverse_dates)):
            date = reverse_dates[index]
            past_date = reverse_dates[index + 1]
            data.loc[data['temporal id'] == date, 'prediction'] = list(
                np.array(data.loc[data['temporal id'] == date, 'prediction']) - np.array(
                    data.loc[data['temporal id'] == past_date, 'train_real_test_prediction']))

            if index == len(reverse_dates) - 2:
                break

    # if target mode is moving average we need to return the target variable to its original state
    elif target_mode == 'moving_average':

        # practical accessible values (target real values in training set and predicted values in test set) will be used 
        # for returning the moving average target (predicted) to original state
        data.loc[:, ('train_real_test_prediction')] = list(training_target['Normal target']) + list(test_prediction)
        data.loc[:, ('prediction')] = list(training_prediction) + list(test_prediction)

        data = data.sort_values(by=['temporal id', 'spatial id'])

        target_granularity = int(target_granularity)

        dates = data['temporal id'].unique()
        for index in range(len(dates)):
            ind = index + target_granularity - 1
            date = dates[ind]
            data.loc[data['temporal id'] == date, 'prediction'] = list(target_granularity * np.array(
                data.loc[data['temporal id'] == date, 'prediction']))

            for i in range(target_granularity - 1):
                past_date = dates[ind - (i + 1)]
                data.loc[
                    data['temporal id'] == date, 'prediction'] = list(np.array(
                    data.loc[data['temporal id'] == date, 'prediction']) - np.array(
                    data.loc[data['temporal id'] == past_date, 'train_real_test_prediction']))
                
            if date in test_dates:
                data.loc[data['temporal id'] == date, 'train_real_test_prediction'] = data.loc[data['temporal id'] == date, 'prediction']
            
            if ind == len(dates) - 1:
                break

    # if target mode is differential we need to return the target variable to its original state
    elif target_mode == 'differential':

        # practical accessible values (target real values in training set and predicted values in test set) will be used 
        # for returning the differential target (predicted) to original state
        data.loc[:, ('train_real_test_prediction')] = list(training_target['Normal target']) + list(test_prediction)
        data.loc[:, ('prediction')] = list(training_prediction) + list(test_prediction)

        data = data.sort_values(by=['temporal id', 'spatial id'])

        dates = data['temporal id'].unique()
        for index in range(len(dates)):
            date = dates[index + 1]
            past_date = dates[index]
            data.loc[
                data['temporal id'] == date, 'prediction'] = list(np.array(
                data.loc[data['temporal id'] == date, 'prediction']) + np.array(
                data.loc[data['temporal id'] == past_date, 'train_real_test_prediction']))
            if date in test_dates:
                data.loc[data['temporal id'] == date, 'train_real_test_prediction'] = list(data.loc[data['temporal id'] == date, 'prediction'])

            if index == len(dates) - 2:
                break

    else:
        raise ValueError("The target_mode is not valid.")
        
    data = data.sort_values(by=['temporal id', 'spatial id'])
    training_set = data[data['type'] == 1]
    test_set = data[data['type'] == 2]

    training_prediction = list(training_set['prediction'])
    test_prediction = list(test_set['prediction'])

    training_target = training_set.drop(['type', 'prediction', 'train_real_test_prediction'], axis=1)
    test_target = test_set.drop(['type', 'prediction', 'train_real_test_prediction'], axis=1)

    if same_train_test == True:
        test_target = training_target
        test_prediction = training_prediction
        
    return training_target, test_target, training_prediction, test_prediction

