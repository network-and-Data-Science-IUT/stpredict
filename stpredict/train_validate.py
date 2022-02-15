import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import sys
    import shutil
    import datetime
    from .models import KNN_REGRESSOR,KNN_CLASSIFIER,NN_REGRESSOR,NN_CLASSIFIER,GLM_REGRESSOR,GLM_CLASSIFIER,GBM_REGRESSOR,GBM_CLASSIFIER
    from sklearn.model_selection import ParameterGrid
    from sklearn.model_selection import KFold
    from multiprocessing import Pool
    from functools import partial
    from tqdm import tqdm
    import random
    import os
    import matplotlib.pyplot as plt
    import datetime
    from .split_data import split_data
    from .performance import performance
    from .scaling import data_scaling
    from .scaling import target_descale
    from .select_features import select_features
    from .train_evaluate import train_evaluate
    from .train_evaluate import inner_train_evaluate
    from .get_normal_target import get_normal_target
    from .get_trivial_values import get_trivial_values
    from .apply_performance_mode import apply_performance_mode
    from .get_target_quantities import get_target_quantities
    from .rank_covariates import rank_covariates
    from .rank_features import rank_features
    from .get_target_temporal_ids import get_target_temporal_ids
    from .configurations import *
    import imp
    import inspect
    import multiprocessing
    from multiprocessing import get_context
    from keras import backend as K

# multiprocessing.set_start_method('spawn', force=True)
K.clear_session()
warnings.filterwarnings("once")


#####################################################################################################

# producing list of parameter values combinations from parameter grid specified by user
def get_parameter_list(model_name, user_params, model_type):

    grid_flag = False
    if model_name == 'nn':

        if 'hidden_layers_structure' in user_params:
            error_msg = 'The value of hidden_layers_structure in NN model parameters must be a list of tuples including number of neurons and activation function of each layer.'
            if not isinstance(user_params['hidden_layers_structure'],list):
                raise ValueError(error_msg)
            elif all([isinstance(item, list) for item in user_params['hidden_layers_structure']]):
                grid_flag = True
                for grid in user_params['hidden_layers_structure']:
                    if not all([isinstance(item, tuple) for item in grid]):
                        raise ValueError(error_msg)
                    elif not all([len(item) == 2 for item in grid]):
                        raise ValueError(error_msg)
            elif all([isinstance(item, tuple) for item in user_params['hidden_layers_structure']]):
                if not all([len(item) == 2 for item in user_params['hidden_layers_structure']]):
                    raise ValueError(error_msg)
            else:raise ValueError(error_msg)
            # remove duplicate information on network structure
            user_params = {key:user_params[key] for key in user_params.keys() if key not in ['hidden_layers_neurons', 'hidden_layers_activations', 'hidden_layers_number']}
            
        else:
            # extract hidden_layers_structure from hidden_layers_neurons, hidden_layers_activations and hidden_layers_number
            if 'hidden_layers_neurons' not in user_params:
                user_params['hidden_layers_neurons'] = [None]
            elif type(user_params['hidden_layers_neurons']) in (np.ndarray,list):
                grid_flag = True
                if not all([isinstance(item , int) for item in user_params['hidden_layers_neurons']]):
                    raise TypeError('The value of hidden_layers_neurons must be of type integer or list of integers.')
            elif type(user_params['hidden_layers_neurons']) == int:
                user_params['hidden_layers_neurons'] = list([user_params['hidden_layers_neurons']])
            else:
                raise TypeError('The value of hidden_layers_neurons must be of type integer or list of integers.')

            if 'hidden_layers_activations' not in user_params:
                user_params['hidden_layers_activations'] = [None]
            elif type(user_params['hidden_layers_activations']) in (np.ndarray,list):
                grid_flag = True
            else:
                user_params['hidden_layers_activations'] = list([user_params['hidden_layers_activations']])

            if 'hidden_layers_number' not in user_params:
                user_params['hidden_layers_number'] = [1]
            elif type(user_params['hidden_layers_number']) in (np.ndarray,list):
                grid_flag = True
                if not all([isinstance(item,int) for item in user_params['hidden_layers_number']]):
                    raise TypeError('The value of hidden_layers_number must be of type integer or list of integers.')
            elif type(user_params['hidden_layers_number']) == int:
                user_params['hidden_layers_number'] = list([user_params['hidden_layers_number']])
            else:
                raise TypeError('The value of hidden_layers_number must be of type integer or list of integers.')

            user_params['hidden_layers_structure'] = []
            for layer_number in user_params['hidden_layers_number']:
                for act_func in user_params['hidden_layers_activations']:
                    for neuron_num in user_params['hidden_layers_neurons']:
                        layer_structure = []
                        for layer in range(1,layer_number+1):
                            layer_structure.append(tuple((neuron_num,act_func)))
                        user_params['hidden_layers_structure'].append(layer_structure)

            # remove duplicate information on network structure
            user_params = {key:user_params[key] for key in user_params.keys() if key 
                           not in ['hidden_layers_neurons', 'hidden_layers_activations', 'hidden_layers_number']}

        # checking other parameters if have multiple values
        for parameter in ['output_activation', 'loss', 'optimizer', 'early_stopping_monitor',
                          'early_stopping_patience', 'batch_size', 'validation_split', 'epochs']:
            if parameter in user_params.keys():
                if type(user_params[parameter]) in (np.ndarray,list):
                    grid_flag = True
                    
        for key in user_params.keys():
            if type(user_params[key]) not in [np.ndarray,list]:
                user_params[key] = list([user_params[key]])

        parameter_list = list(ParameterGrid(user_params))

    if model_name == 'knn':

        for parameter in ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p',
                              'metric', 'metric_params', 'n_jobs']:
            if parameter in user_params.keys():
                if type(user_params[parameter]) in (np.ndarray,list):
                    grid_flag = True
        
        for key in user_params.keys():
            if type(user_params[key]) not in [np.ndarray,list]:
                user_params[key] = list([user_params[key]])

        parameter_list = list(ParameterGrid(user_params))

    if model_name == 'gbm':
        if model_type == 'regression':
            parameters = ['loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion', 'min_samples_split',
                          'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'min_impurity_decrease',
                          'init', 'random_state', 'max_features', 'alpha', 'verbose', 'max_leaf_nodes',
                          'warm_start', 'validation_fraction', 'n_iter_no_change', 'tol', 'ccp_alpha']

        elif model_type == 'classification':
            parameters = ['loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion', 'min_samples_split',
                          'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'min_impurity_decrease',
                          'init', 'random_state', 'max_features', 'verbose', 'max_leaf_nodes',
                          'warm_start', 'validation_fraction', 'n_iter_no_change', 'tol', 'ccp_alpha']

        for parameter in parameters:
            if parameter in user_params.keys():
                if type(user_params[parameter]) in (np.ndarray,list):
                    grid_flag = True
        
        for key in user_params.keys():
            if type(user_params[key]) not in [np.ndarray,list]:
                user_params[key] = list([user_params[key]])

        parameter_list = list(ParameterGrid(user_params))

    if model_name == 'glm':

        if model_type == 'regression':
            parameters = ['alpha', 'l1_ratio', 'fit_intercept', 'normalize', 'precompute', 'max_iter', 'copy_X', 'tol',
                          'warm_start', 'positive', 'random_state', 'selection']

        elif model_type == 'classification':
            parameters = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state',
                          'solver', 'max_iter', 'multi_class', 'verbose', 'warm_start', 'n_jobs', 'l1_ratio']

        for parameter in parameters:
            if parameter in user_params.keys():
                if type(user_params[parameter]) in (np.ndarray,list):
                    grid_flag = True

        
        for key in user_params.keys():
            if type(user_params[key]) not in [np.ndarray,list]:
                user_params[key] = list([user_params[key]])

        parameter_list = list(ParameterGrid(user_params))
        
        
    return parameter_list

# reporting the performance of models in csv file
def report_performance(errors_dict, max_history, ordered_covariates_or_features,
                       feature_sets_indices, performance_measures, feature_selection_type,
                       models_name_list, models_best_parameters_number, forecast_horizon,
                       data_temporal_size, report_type):
    
    output_data_frame = pd.DataFrame(columns = ['model name', 'history length', 'feature or covariate set'] + performance_measures)
    
    for model_name in models_name_list:
        for history in range(1, max_history+1):
            
            feature_sets = {feature_set_number:[] for feature_set_number in range(len(feature_sets_indices[history-1]))}
            for feature_set_number in range(len(feature_sets_indices[history-1])):
                # get the name of features in a current selected feature set which are in
                # a format 'temporal covariate x' or 'spatial covariate x'
                for index in feature_sets_indices[history-1][feature_set_number]:
                    feature_original_name = ordered_covariates_or_features[history-1][index]
                    if feature_original_name.endswith(' t+'):
                        feature_original_name = feature_original_name.split(' t+')[0]
                    if feature_selection_type == 'covariate' and len(feature_original_name.split(' '))>1:
                        feature_original_name = feature_original_name.split(' ')[0]
                        
                    feature_sets[feature_set_number].append(feature_original_name)

            temp = pd.DataFrame(columns = ['model name', 'history length', 'feature or covariate set'] + list(performance_measures))
            temp.loc[:,('feature or covariate set')] = list([' , '.join(feature_sets[feature_set_number]) for feature_set_number in range(len(feature_sets_indices[history-1]))])
            temp.loc[:,('model name')] = model_name
            temp.loc[:,('history length')] = history
            
            for measure in performance_measures:
                errors_list = []
                for feature_set_number in range(len(feature_sets_indices[history-1])):
                    model_best_parameters_number = models_best_parameters_number[model_name][(history, feature_set_number)]
                    errors_list.append(errors_dict[measure][model_name][(history, feature_set_number, model_best_parameters_number)])
                temp.loc[:,(measure)] = list(errors_list)
            
            output_data_frame = output_data_frame.append(temp)
    
    address = './performance/validation process/'
    if os.path.exists(address):
        files = [f for f in os.listdir(address) if ((os.path.isfile(os.path.join(address, f))) and (report_type in f))]
        
        if len(files) == 0:
            output_data_frame.to_csv('{0}{1} performance report forecast horizon = {2}.csv'.format(address, report_type, forecast_horizon), index = False)
        elif len(files) == 1:
            os.rename(address+files[0], '{0}{1} performance report forecast horizon = {2}, test-point #1.csv'.format(address, report_type, forecast_horizon))
            output_data_frame.to_csv('{0}{1} performance report forecast horizon = {2}, test-point #2.csv'.format(address, report_type, forecast_horizon), index = False)
        else:
            output_data_frame.to_csv('{0}{1} performance report forecast horizon = {2}, test-point #{3}.csv'.format(address, report_type, forecast_horizon, len(files)+1), index = False)
            
    else:
        os.makedirs(address)
        output_data_frame.to_csv('{0}{1} performance report forecast horizon = {2}.csv'.format(address, report_type, forecast_horizon), index = False)
    
#############################################################

def parallel_run(prediction_arguments):
    train_predictions, validation_predictions, trained_model, number_of_parameters = inner_train_evaluate(training_data = prediction_arguments[0],
                                                                              validation_data = prediction_arguments[1],
                                                                              model = prediction_arguments[2], 
                                                                              model_type = prediction_arguments[3],
                                                                              model_parameters = prediction_arguments[4],
                                                                              labels = prediction_arguments[5],
                                                                              verbose = prediction_arguments[6])
    return train_predictions, validation_predictions, number_of_parameters

    


def save_prediction_data_frame(models_name_list, target_real_values, fold_validation_predictions,
                               fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                               models_best_parameters_number, forecast_horizon, data_temporal_size, prediction_type, model_type, labels):
    
    fold_number = 1
    
    try:
        labels = [int(item) for item in labels]
    except:
        pass
    
    if model_type == 'regression':
        prediction_data_frame = pd.DataFrame(columns = ['model name', 'spatial id', 'temporal id', 'real', 'prediction'])
    elif model_type == 'classification':
        prediction_data_frame = pd.DataFrame(columns = ['model name', 'spatial id', 'temporal id', 'real'] + ['class '+str(item) for item in labels])
    
    for model_number, model_name in enumerate(models_name_list):

        model_best_history_length = models_best_history_length[model_name]
        model_best_feature_set_number = models_best_feature_set_number[model_name]
        model_best_parameters_number = models_best_parameters_number[model_name][(model_best_history_length, model_best_feature_set_number)]
        
        # if there was a problem in finding best_error
        if model_best_history_length is None:
            model_best_history_length = 1
            model_best_feature_set_number = 0
        
        if prediction_type == 'validation':
            
            temp = target_real_values['validation'][(fold_number, model_best_history_length)]
            if model_type == 'regression':
                temp.loc[:,('prediction')] = fold_validation_predictions[model_name][(fold_number, model_best_history_length, model_best_feature_set_number, model_best_parameters_number)]
            elif model_type == 'classification':
                for label_number, label_name in enumerate(labels):
                    temp.loc[:,('class '+str(label_name))] = list(fold_validation_predictions[model_name][(fold_number, model_best_history_length, model_best_feature_set_number, model_best_parameters_number)][:,label_number])
        
        elif prediction_type == 'training':
            
            temp = target_real_values['training'][(fold_number, model_best_history_length)]
            if model_type == 'regression':
                temp.loc[:,('prediction')] = fold_training_predictions[model_name][(fold_number, model_best_history_length, model_best_feature_set_number, model_best_parameters_number)]
            elif model_type == 'classification':
                for label_number, label_name in enumerate(labels):
                    temp.loc[:,('class '+str(label_name))] = list(fold_training_predictions[model_name][(fold_number, model_best_history_length, model_best_feature_set_number, model_best_parameters_number)][:,label_number])
        
        
        temp.loc[:,('model name')] = model_name
        temp = temp.drop(['Target'], axis = 1)
        temp.rename(columns = {'Normal target':'real'}, inplace = True)
        if model_type == 'regression':
            temp = temp[['model name', 'spatial id', 'temporal id', 'real', 'prediction']]
        elif model_type == 'classification':
            temp = temp[['model name', 'spatial id', 'temporal id', 'real'] + ['class '+str(item) for item in labels]]

        prediction_data_frame = prediction_data_frame.append(temp)
    
    # prediction_data_frame = prediction_data_frame.rename(columns = {'temporal id':'predictive time point'})
    address = './prediction/validation process/'
    
    if os.path.exists(address):
        files = [f for f in os.listdir(address) if ((os.path.isfile(os.path.join(address, f))) and (prediction_type in f))]
        if len(files) == 0:
            prediction_data_frame.to_csv('{0}{1} prediction forecast horizon = {2}.csv'.format(address, prediction_type, forecast_horizon), index = False)
        elif len(files) == 1:
            os.rename(address+files[0], '{0}{1} prediction forecast horizon = {2}, test-point #1.csv'.format(address, prediction_type, forecast_horizon))
            prediction_data_frame.to_csv('{0}{1} prediction forecast horizon = {2}, test-point #2.csv'.format(address, prediction_type, forecast_horizon), index = False)
        else:
            prediction_data_frame.to_csv('{0}{1} prediction forecast horizon = {2}, test-point #{3}.csv'.format(address, prediction_type, forecast_horizon, len(files)+1), index = False)
            
    else:
        os.makedirs(address)
        prediction_data_frame.to_csv('{0}{1} prediction forecast horizon = {2}.csv'.format(address, prediction_type, forecast_horizon), index = False)
    
###########################################################################################

def save_ranking(ranked_features_to_save, feature_selection_type, ordered_covariates_or_features):
    
    if feature_selection_type == 'covariate':
        i = 1
        rank = []
        ranked_df = pd.DataFrame(columns = ['rank','feature'])
        ranked_df['feature'] = list(ranked_features_to_save)
        for feature in ranked_features_to_save:
            if ordered_covariates_or_features[0][i-1] in feature:
                rank.append(i)
            else:
                i = i+1
                rank.append(i)
            if i>len(ordered_covariates_or_features[0]):break
        ranked_df['rank'] = rank
        
    else:
        ranked_df = pd.DataFrame(ordered_covariates_or_features).T
        ranked_df.columns = ['history #'+str(h)+' features' for h in range(1,len(ordered_covariates_or_features)+1)]
        ranked_df = ranked_df.reset_index()
        ranked_df = ranked_df.rename(columns = {'index':'rank'})
        ranked_df['rank'] = ranked_df['rank'] + 1
        
    address = './ranked features/'
    
    if os.path.exists('ranked features'):
        files = [f for f in os.listdir(address) if os.path.isfile(os.path.join(address, f))]
        if len(files) == 1:
            os.rename(address+files[0], address+'ranked features for test point #1.csv')
            ranked_df.to_csv(address+'ranked features for test point #2.csv', index = False)
        else:
            ranked_df.to_csv(address+f'ranked features for test point #{len(files)+1}.csv', index = False)
    else:
        os.mkdir('./ranked features/')
        ranked_df.to_csv(address+'ranked features.csv', index = False)
            
    
    
def train_validate(data, feature_sets, forced_covariates = [], instance_validation_size = 0.3,
                   instance_testing_size = 0,
                   fold_total_number = 5, instance_random_partitioning = False,
                   forecast_horizon = 1, models = ['knn'], mixed_models = None,  model_type = 'regression',
                   splitting_type = 'training-validation',
                   performance_measures = None, performance_benchmark = None, performance_mode = 'normal', 
                   feature_scaler = None, target_scaler = None, labels = None, performance_report = True,
                   save_predictions = True, save_ranked_features = True, verbose = 0):
    
    supported_models_name = ['nn', 'knn', 'glm', 'gbm']
    supported_performance_measures = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score', 'AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']
    models_list = [] # list of models (str or callable)
    models_parameter_list = [] # list of models' parameters (dict or None)
    models_name_list = [] # list of models' names (str)
    base_models_name_list = [] # list of base models' names (str)
    mixed_models_name_list = [] # list of base models' names (str)
    same_train_validation_sets = False # Flag to specify whether or not both training and validation must be performed based on same set
    ranked_features_to_save = [] # list of ranked features to be saved
    
    ############################ reading and validating inputs
    
    ############## forecast horizon
    
    if (type(forecast_horizon) != int) or (forecast_horizon<=0):
        raise ValueError("The forecast_horizon input must be an integer greater than 0.")
    
    ############## data input
    
    data_list = []
    if type(data) == list:
        for history in range(1,len(data)+1):
            if type(data[history-1]) == pd.DataFrame:
                data_list.append(data[history-1])
            elif type(data[history-1]) == str:
                try:
                    data_list.append(pd.read_csv(data[history-1]))
                except FileNotFoundError:
                    raise FileNotFoundError("File '{0}' does not exist.".format(data[history-1]))
            else:
                raise ValueError("The input data must be a list of DataFrames or strings of data addresses.")
    else:
        raise ValueError("The input data must be a list of DataFrames or strings of data addresses.")
        
    # find the target mode, target granularity, and granularity by decoding target variable column name
    # and get the target temporal id from temporal id
    granularity = [1]*len(data_list)
    target_granularity = [1]*len(data_list)
    target_mode = ['normal']*len(data_list)
    for i in range(len(data_list)):
        target_mode[i], target_granularity[i], granularity[i], data_list[i] = get_target_quantities(data_list[i])
        if 'target temporal id' in data_list[i].columns:
            data_list[i] = data_list[i].rename(columns={'target temporal id':'temporal id'})
        else:
            data_list[i], _ = get_target_temporal_ids(temporal_data = data_list[i], forecast_horizon = forecast_horizon,
                                                   granularity = granularity[i])
        temp_data = data_list[i].sort_values(by = ['temporal id','spatial id']).copy()
        number_of_spatial_units = len(temp_data['spatial id'].unique())
        if all(temp_data.tail(granularity[i]*forecast_horizon*number_of_spatial_units)['Target'].isna()):
            data_list[i] = temp_data.iloc[:-(granularity[i]*forecast_horizon*number_of_spatial_units)]
        
    
    max_history = len(data_list)
        
    ############## models input
    
    if type(models) != list:
        raise TypeError("The models input must be of type list.")
        
    number_of_user_defined_models = 0
    
    for item in models:
        
        # if the item is the dictionary of model name and its parameters
        if type(item) == dict:       
            model_name = list(item.keys())[0]
            
            # if the dictionary contain only one of the supported models
            if len(item) == 1:
                
                if model_name in supported_models_name:
                
                    # if model is not duplicate 
                    if model_name not in models_list:
                        models_list.append(model_name)
                        models_name_list.append(model_name)
                        base_models_name_list.append(model_name)
                        # if the value of the model name is dictionary of models parameter list
                        if type(item[model_name]) == dict:
                            # get list of different combinations of model parameters
                            model_parameters = get_parameter_list(model_name = model_name, user_params = item[model_name],\
                                                                       model_type = model_type)
                            models_parameter_list.append(model_parameters)
                        else:
                            models_parameter_list.append([None])
                            print("\nWarning: The values in the dictionary items of models list must be a dictionary of the model hyper parameter names and values. Other values will be ignored.\n")
                    else:
                        models.remove(item)
                        print("\nWarning: Some of the predefined models are mentioned in the models' input multiple times. The duplicate cases will be ignored.\n")
                else:
                    models.remove(item)
                    print("\nWarning: The keys in the dictionary items of models list must be one of the supported model names. The incompatible cases will be ignored.\n")
            else:
                models[models.index(item)] = {list(item.keys())[0]:list(item.values())[0]}
                print("\nWarning: Each dictionary item in models list must contain only one item with a name of one of the supported models as a key and the parameters of that model as value. The incompatible cases will be ignored.\n")
        
        # if the item is only name of model whithout parameters
        elif type(item) == str:
            if (item in supported_models_name):
                if (item not in models_list):
                    models_list.append(item)
                    models_name_list.append(item)
                    base_models_name_list.append(item)
                    models_parameter_list.append([None])
            else:
                models.remove(item)
                print("\nWarning: The string items in the models list must be one of the supported model names. The incompatible cases will be ignored.\n")
        
        # if the item is user defined function
        elif callable(item):
            if item.__name__ in supported_models_name:
                raise Exception("User-defined model names must be different from predefined models:['knn', 'glm', 'gbm', 'nn']")
            if item.__name__ in models_name_list:
                raise Exception("User-defined models can not have the same names.")
            models_list.append(item)
            models_name_list.append(item.__name__)
            base_models_name_list.append(item.__name__)
            models_parameter_list.append([None])
            number_of_user_defined_models += 1
            
            callable_source_code = inspect.getsource(item)
            mode = "a" if number_of_user_defined_models>1 else "w"
            user_defined_models = open(imp.find_module('stpredict')[1]+"/user_defined_models.py",mode)
            for line in callable_source_code:
                user_defined_models.write(line)
            user_defined_models.close()
                        
        else:
            models.remove(item)
            print("\nWarning: The items in the models list must be of type string, dict or callable. The incompatible cases will be ignored.\n")
    
    if len(models_list) < 1:
        raise ValueError("There is no item in the models list or the items are invalid.")

    ############## mixed models input
    
    if type(mixed_models) != list:
        raise TypeError("The mixed_models input must be of type list.")
    
    for item in mixed_models:
        
        # if the item is the dictionary of model name and its parameters
        if type(item) == dict:       
            model_name = list(item.keys())[0]
            
            # if the dictionary contain only one of the supported models
            if len(item) == 1:
                
                if model_name in supported_models_name:
                    # if model is not duplicate 
                    if 'mixed_' + model_name not in mixed_models_name_list:
                        models_list.append('mixed_'+model_name)
                        models_name_list.append('mixed_' + model_name)
                        mixed_models_name_list.append('mixed_' + model_name)
                        # if the value of the model name is dictionary of models parameter list
                        if type(item[model_name]) == dict:
                            # get list of different combinations of model parameters
                            model_parameters = get_parameter_list(model_name = model_name, user_params = item[model_name],\
                                                                       model_type = model_type)
                            models_parameter_list.append(model_parameters)
                        else:
                            models_parameter_list.append([None])
                            print("\nWarning: The values in the dictionary items of mixed_models list must be a dictionary of the model hyper parameter names and values. Other values will be ignored.\n")
                    else:
                        mixed_models.remove(item)
                        print("\nWarning: Some of the predefined models are mentioned in the mixed_models list multiple times. The duplicate cases will be ignored.\n")
                else:
                    mixed_models.remove(item)
                    print("\nWarning: The keys in the dictionary items of mixed_models list must be one of the supported model names. The incompatible cases will be ignored.\n")
            else:
                mixed_models[mixed_models.index(item)] = {list(item.keys())[0]:list(item.values())[0]}
                print("\nWarning: Each dictionary item in mixed_models list must contain only one item with a name of one of the supported models as a key and the parameters of that model as value. The incompatible cases will be ignored.\n")
        
        # if the item is only name of model whithout parameters
        elif type(item) == str:
            if (item in supported_models_name):
                if ('mixed_'+item not in mixed_models_name_list):
                    models_list.append('mixed_'+item)
                    models_name_list.append('mixed_'+item)
                    mixed_models_name_list.append('mixed_'+item)
                    models_parameter_list.append([None])
            else:
                mixed_models.remove(item)
                print("\nWarning: The string items in the mixed_models list must be one of the supported model names. The incompatible cases will be ignored.\n")
        
        # if the item is user defined function
        elif callable(item):
            if item.__name__ in supported_models_name:
                raise Exception("User-defined model names must be different from predefined models:['knn', 'glm', 'gbm', 'nn']")
            if item.__name__ in models_name_list:
                raise Exception("User-defined models can not have the same names.")
            models_list.append(item)
            models_name_list.append(item.__name__)
            mixed_models_name_list.append(item.__name__)
            models_parameter_list.append([None])
            number_of_user_defined_models += 1
            
            callable_source_code = inspect.getsource(item)
            mode = "a" if number_of_user_defined_models>1 else "w"
            user_defined_models = open(imp.find_module('stpredict')[1]+"/user_defined_models.py",mode)
            for line in callable_source_code:
                user_defined_models.write(line)
            user_defined_models.close()
                        
        else:
            mixed_models.remove(item)
            print("\nWarning: The items in the mixed_models list must be of type string, dict or callable. The incompatible cases will be ignored.\n")
        
        
    ############## performance measure input
    
    if (type(performance_measures) != list) and (performance_measures is not None):
        raise TypeError("The performance_measures must be of type list.")
        
    zero_encounter_flag = 0
    for history in range(1,max_history+1):
        data = data_list[history-1].copy()
        if len(data[data['Target']==0]) > 0: zero_encounter_flag = 1
        
    if performance_measures is None:
        if model_type == 'regression':
            if zero_encounter_flag == 0:
                performance_measures = ['MAPE']
            else: performance_measures = ['MAE']
        else:
            performance_measures = ['AUC']
        
    unsupported_measures = list(set(performance_measures)-set(supported_performance_measures))
    if len(unsupported_measures) > 0:
        print("\nWarning: Some of the specified measures are not valid:\n{0}\nThe supported measures are: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score', 'AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']\n".format(unsupported_measures))
    
    performance_measures = list(set([measure for measure in supported_performance_measures if measure in performance_measures]))
    
    if (splitting_type == 'cross-validation') and ('MASE' in performance_measures):
        performance_measures.remove('MASE')
        print("\nWarning: The 'MASE' measure cannot be measured in cross-validation splitting mode.\n")
    
    if (splitting_type != 'cross-validation') and (instance_random_partitioning == True) and ('MASE' in performance_measures):
        performance_measures.remove('MASE')
        print("\nWarning: The 'MASE' measure cannot be measured in random partitioning mode.\n")
        
    
    if (model_type == 'regression') and (any([item in ['likelihood', 'AUC', 'AUPR', 'AIC', 'BIC'] for item in performance_measures])):
        performance_measures = [item for item in performance_measures if item not in ['likelihood', 'AUC', 'AUPR', 'AIC', 'BIC']]
        print("\nWarning: Some of the measures in the performance_measures can not be measured for {0} task and will be ignored.\n".format(model_type))
    
    if (model_type == 'classification') and (any([item in ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score'] for item in performance_measures])):
        performance_measures = [item for item in performance_measures if item not in ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']]
        print("\nWarning: Some of the measures in the performance_measures can not be measured for {0} task and will be ignored.\n".format(model_type))
    
    if len(performance_measures) < 1:
        raise ValueError("No valid measure is specified.")
        
    ############## performance_benchmark input
    
    if (performance_benchmark not in supported_performance_measures) or (performance_benchmark is None):
        if model_type == 'regression':
            if zero_encounter_flag == 0:
                performance_benchmark = 'MAPE'
            else:
                performance_benchmark = 'MAE'
        else:
            performance_benchmark = 'AUC'
        if performance_benchmark is not None:
            print("\nWarning: The specified performance_benchmark must be one of the supported performance measures: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score', 'AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']\nThe incompatible cases will be ignored and replaced with '{0}'.\n".format(performance_benchmark))
            
    if (performance_benchmark in ['likelihood', 'AUC', 'AUPR', 'AIC', 'BIC']) and (model_type == 'regression'): 
        if zero_encounter_flag == 0:
            performance_benchmark = 'MAPE'
        else:
            performance_benchmark = 'MAE'
        print("\nWarning: The specified performance_benchmark can not be measured for regression models. Thus the performance_benchmark will be set to '{0}'.\n".format(performance_benchmark))
    
    if (performance_benchmark in ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']) and (model_type == 'classification'):
        performance_benchmark = 'AUC'
        print("\nWarning: The specified performance_benchmark can not be measured for classification models. Thus the performance_benchmark will be set to '{0}'.\n".format(performance_benchmark))
            
#     if (performance_benchmark == 'MAPE') and (zero_encounter_flag == 1):
#         performance_benchmark = 'MAE'
#         print("\nWarning: The input data contain some zero values for Target variable. Therefore 'MAPE' can not be used as a benchmark and the benchmark will be set to 'MAE'.\n")
    
    if number_of_user_defined_models>0 or any([model not in ['glm', 'mixed_glm'] for model in models_name_list]):
        if (performance_benchmark == 'AIC') or (performance_benchmark == 'BIC'):
            if model_type == 'classification':
                performance_benchmark = 'AUC'
            elif model_type == 'regression':
                if zero_encounter_flag == 0:
                    performance_benchmark = 'MAPE'
                else:
                    performance_benchmark = 'MAE'
            print("\nWarning: The 'AIC' and 'BIC' measures can only be measured for classification glm model. Thus the performance_benchmark will be set to '{0}'.\n".format(performance_benchmark))
    
    if (splitting_type == 'cross-validation') and (performance_benchmark == 'MASE'):
        # if zero_encounter_flag == 0:
        #     performance_benchmark = 'MAPE'
        # else:
        #     performance_benchmark = 'MAE'
        raise Exception("Error: The 'MASE' measure cannot be measured in cross-validation splitting mode.")

    if (splitting_type != 'cross-validation') and (instance_random_partitioning == True) and (performance_benchmark == 'MASE'):
        # if zero_encounter_flag == 0:
        #     performance_benchmark = 'MAPE'
        # else:
        #     performance_benchmark = 'MAE'
        raise Exception("Error: The 'MASE' measure cannot be measured in random partitioning mode.")          
        
    if performance_benchmark not in performance_measures:
        performance_measures.append(performance_benchmark)

    
    ############## model_type input and labels
    
    if model_type == 'classification':
        if labels is None:
            labels = list(data_list[0]['Normal target'].unique())
            labels.sort()
        elif type(labels) != list:
            raise TypeError("The labels input must be of type list.")
        elif any([len(set(data['Normal target'].dropna().unique())-set(labels))>0 for data in data_list]):
            raise ValueError("Some of the class labels in the input data are not in labels input.")
        
        if performance_mode != 'normal':
            if performance_mode is not None:
                print("\nWarning: The 'cumulative' or 'moving average' performance_mode can not be used for the classification.")
            performance_mode = 'normal'
        if target_scaler is not None:
            target_scaler = None
            print("\nWarning: Target scaling can not be performed for the classification.")
        if not all([item == 'normal' for item in target_mode]):
            target_mode = ['normal']*len(target_mode)
            
    elif model_type == 'regression':
        labels = []
    else:
        raise ValueError("The specified model_type is not valid. The supported values are 'regression' and 'classification'.")
    
    
    ############## feature_sets, forced_covariates
    
    if (type(feature_sets) != dict) or (len(feature_sets) > 1):
        raise TypeError("The feature_sets input must be of type dictionary with only one item.")
        
    feature_selection_type = list(feature_sets.keys())[0]
    ranking_method = list(feature_sets.values())[0]
    if feature_selection_type not in ['covariate','feature']:
        raise ValueError("The key of the feature_sets dictionary must be 'covariate' or 'feature'.")
    if ranking_method not in RANKING_METHODS:
        raise ValueError(f"The value of the item in the feature_sets dictionary must be from the list {RANKING_METHODS}.")
    
    if type(forced_covariates) != list:
        raise TypeError("The forced_covariates input must be of type list.")
        
    ############## splitting_type, fold_total_number, instance_testing_size, and instance_validation_size inputs
    
    # check validity of fold_total_number
    if splitting_type == 'cross-validation':

        if fold_total_number is None:
            raise Exception("if the splitting_type is 'cross-validation', the fold_total_number must be specified.")
        if (type(fold_total_number) != int) or (fold_total_number <= 1):
            raise ValueError("The fold_total_number must be an integer greater than 1.")

    # check validity of instance_validation_size
    elif splitting_type == 'training-validation':
    
        if type(instance_validation_size) == float:
            if instance_validation_size > 1:
                raise ValueError("The float instance_validation_size will be interpreted to the proportion of data which is considered as validation set and must be less than 1.")
                
        elif (type(instance_validation_size) != int):
            raise TypeError("The type of instance_validation_size must be int or float.")
        
        # if instance_validation_size is zero the validation of models are performed based on training set
        if instance_validation_size == 0:
            same_train_validation_sets = True
    else:
        raise ValueError("The specified splitting_type is ambiguous. The supported values are 'training-validation' and 'cross-validation'.")
    
    # check validity of instance_testing_size
    if type(instance_testing_size) == float:
        if instance_testing_size > 1:
            raise ValueError("The float instance_testing_size will be interpreted to the proportion of data that is considered as the test set and must be less than 1.")
    elif type(instance_testing_size) != int:
        raise TypeError("The type of instance_testing_size must be int or float.")
    
    if type(instance_random_partitioning) != bool:
        raise TypeError("instance_random_partitioning must be type boolean.")
    
    # for non cross val splitting_type, the fold_total_number  will be set to 1, to perform the prediction process only one time
    if splitting_type != 'cross-validation':
        fold_total_number = 1
    
    # setting the splitting_type of split_data function according to user specified splitting_type in train_validate function
    if splitting_type == 'cross-validation':
        split_data_splitting_type = 'fold'
    else:
        split_data_splitting_type = 'instance'
        
    ############## feature and target scaler, performance mode, performance_report, save_predictions, verbose
    
    if target_scaler not in ['logarithmic', 'normalize', 'standardize', None]:
        raise ValueError("The target_scaler input is not valid. The supported values are 'logarithmic', 'normalize', 'standardize', or None for no scaling.")
    if feature_scaler not in ['logarithmic', 'normalize', 'standardize', None]:
        raise ValueError("The feature_scaler input is not valid. The supported values are 'logarithmic', 'normalize', 'standardize', or None for no scaling.")
    
    if not any(performance_mode.startswith(item) for item in ['normal', 'cumulative', 'moving average']):
        raise ValueError("The performance_mode input is not valid.")
        
    if type(performance_report) != bool:
        raise TypeError("performance_report must be type boolean.")
        
    if type(save_predictions) != bool:
        raise TypeError("save_predictions must be type boolean.")
    
    if type(verbose) != int:
        raise TypeError("verbose must be of type int.")
        
    ############## check the possibility of data deficiency
    
    if (performance_benchmark == 'MASE') and (splitting_type != 'cross-validation'):
        for index,data in enumerate(data_list):
            temp_train, temp_val , temp_test , _ = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                                      instance_validation_size = instance_validation_size, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                                      instance_random_partitioning = instance_random_partitioning, granularity = granularity[index], verbose = 0)
            if len(temp_train['temporal id'].unique()) < (granularity[index]*forecast_horizon) + 1:
                # if zero_encounter_flag == 0:
                #     performance_benchmark = 'MAPE'
                # else:
                #     performance_benchmark = 'MAE'
                raise Exception("Error: There are not enough temporal units in the data to measure 'MASE'.")
            
    
    #################################################### initializing
        
    models_best_history_length = {model_name : None for model_name in models_name_list} # best_history_length for each model
    models_best_feature_set_number = {model_name : None for model_name in models_name_list} # index of best_feature_set in feature_set_indices for each model
    models_best_parameters_number = {model_name : {} for model_name in models_name_list} # index of best_parameters in model_parameters list for each model for each history and featureset
    best_model = None # overall best model
        
    ####### the outputs of running the models 
    fold_training_predictions = {model_name : {} for model_name in models_name_list} # train prediction result for each fold
    fold_validation_predictions = {model_name : {} for model_name in models_name_list} # validation prediction result for each fold
    
    normal_fold_training_predictions = {model_name : {} for model_name in models_name_list} # train prediction result for each fold which is modified to be corresponding to normal target mode values
    normal_fold_validation_predictions = {model_name : {} for model_name in models_name_list} # validation prediction result for each fold which is modified to be corresponding to normal target mode values
    
    performance_fold_training_predictions = {model_name : {} for model_name in models_name_list} # train prediction result for each fold modified with performance_mode to measure performance
    performance_fold_validation_predictions = {model_name : {} for model_name in models_name_list} # validation prediction result for each fold modified with performance_mode to measure performance
    
    number_of_parameters = {model_name : {} for model_name in models_name_list} # number of parameters of the trained model on each fold (needed for 'AIC' and 'BIC')
    
    # training and validation target real values for different history lengths and fold number
    target_real_values = {'training':{},'validation':{}}
    normal_target_real_values = {'training':{},'validation':{}}
    performance_target_real_values = {'training':{},'validation':{}}

    # validation and training error of different measures for each model
    validation_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measures}
    training_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measures}
    
    # set the appropriate min error based on performance_benchmark measure
    if performance_benchmark in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
        overall_min_validation_error = float('Inf')
        models_min_validation_error = {model_name : float('Inf') for model_name in models_name_list}
    else:
        overall_min_validation_error = float('-Inf')
        models_min_validation_error = {model_name : float('-Inf') for model_name in models_name_list}    
    
    # dictionary to hold the min error resulted from testing different parameters for each model and for each history and feature set
    model_parameters_min_validation_error = {model_name : {} for model_name in models_name_list}
    
    knn_alert_flag = 0
    Number_of_cpu = multiprocessing.cpu_count()
    
    #################################################### main part
    
    ########################## ranking
    
    ordered_covariates_or_features = []
    
    for history in range(1,max_history+1):
        
        if feature_selection_type == 'feature':
            data = data_list[history-1].copy()
        else:
            data = data_list[0].copy()
        
        # separating the test part
        raw_train_data, _ , raw_testing_data , _ = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                                  instance_validation_size = None, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                                  instance_random_partitioning = instance_random_partitioning, granularity = granularity[history-1], verbose = 0)
        if history == 1:
            number_of_temporal_units = len(raw_train_data['temporal id'].unique())
            
        if feature_selection_type == 'feature':
            ordered_covariates_or_features.append(rank_features(data=raw_train_data.copy(),
                                                                    ranking_method=ranking_method,
                                                                    forced_covariates=forced_covariates))
        else:
            ordered_covariates_or_features.append(rank_covariates(data=raw_train_data.copy(),
                                                                    ranking_method=ranking_method,
                                                                    forced_covariates=forced_covariates))
            
    feature_sets_indices = [] # feature_set_indices set of all history lengths
    for history in range(max_history):
        history_feature_sets_indices = [] # feature_set_indices for specific history length

        # the start point for considering number of features or covariates in feature set indices
        # if futuristic covariates exist in the list, the start point will set in a way to always
        # consider forced covariates in the index
        start_point = 0
        for feature in ordered_covariates_or_features[history]:
            if len(feature.split(' '))>1:
                if feature.split(' ')[0] in forced_covariates:
                    start_point +=1
            else:
                if feature in forced_covariates:
                    start_point +=1
                    
        if start_point == 0 : start_point = 1
        for number_of_features in range(start_point,len(ordered_covariates_or_features[history])+1):
            history_feature_sets_indices.append(list(range(number_of_features)))
        feature_sets_indices.append(history_feature_sets_indices)
        
    # set the appropriate min error based on performance_benchmark measure
    if performance_benchmark in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
        for model_name in models_name_list:
            for history in range(1,max_history+1):
                for feature_set_number in range(len(feature_sets_indices[history-1])):
                    model_parameters_min_validation_error[model_name][(history,feature_set_number)] = float('Inf')
    else:
        for model_name in models_name_list:
            for history in range(1,max_history+1):
                for feature_set_number in range(len(feature_sets_indices[history-1])):
                    model_parameters_min_validation_error[model_name][(history,feature_set_number)] = float('-Inf')
    
    ########################## (loop over history_length, feature_sets_indices, and folds)
    
    for history in range(1,max_history+1):
        
        if verbose>0:
            print("\n"+"-"*55+"\nValidation process is running for history length = {0}.\n".format(history)+"-"*55+"\n")
        
        # get the data with specific history length
        data = data_list[history-1].copy()
        
        # separating the test part
        raw_train_data, _ , raw_testing_data , _ = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                                  instance_validation_size = None, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                                  instance_random_partitioning = instance_random_partitioning, granularity = granularity[history-1], verbose = 0)

        ####################### Running base models
        
        if len(mixed_models_name_list)>0:
            print('\nBase models are running.\n')
        
        with get_context("spawn").Pool(processes = Number_of_cpu) as prediction_pool:
            
            # initializing the pool for parallel run
            pool_list = [] # list of all the different combination of the arguments of pool function

            for feature_set_number in range(len(feature_sets_indices[history-1])):

                indices = feature_sets_indices[history-1][feature_set_number]
                names_to_select = [ordered_covariates_or_features[history-1][index] for index in indices]

                # select the features
                train_data = select_features(data = raw_train_data.copy(), ordered_covariates_or_features = names_to_select)
                
                for model_number, model in enumerate(models_list):

                    model_name = models_name_list[model_number]
                    if model_name in base_models_name_list:

                        for model_parameters_number in range(len(models_parameter_list[model_number])):
                            model_parameters = models_parameter_list[model_number][model_parameters_number]

                            for fold_number in range(1, fold_total_number + 1):
                                
                                # get the current fold training and validation data
                                training_data, validation_data, _ , _ = split_data(data = train_data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = None,
                                                              instance_validation_size = instance_validation_size, fold_total_number = fold_total_number, fold_number = fold_number,
                                                              splitting_type = split_data_splitting_type, instance_random_partitioning = instance_random_partitioning, 
                                                              granularity = granularity[history-1], verbose = 0)
                                
                                if same_train_validation_sets == True:
                                    validation_data = training_data.copy()
                                    
                                if model_parameters is not None:
                                    if 'n_neighbors' in model_parameters.keys():
                                        if type(model_parameters['n_neighbors']) == int:
                                            if (model_parameters['n_neighbors']>len(training_data)) and (knn_alert_flag == 0):
                                                print("\nWarning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
                                                print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]\n")
                                                knn_alert_flag = 1

                                # saving the target real values of each fold for data with different history lengths to use in
                                # calculation of performance and saving real and predicted target values in csv files
                                needed_columns = ['spatial id', 'temporal id', 'Target', 'Normal target']

                                # the target real values dataframe is copied for each combination of fold_number, history, feature_set_number , and model_parameters_number
                                target_real_values['training'][(fold_number, history)] = training_data[needed_columns]
                                target_real_values['validation'][(fold_number, history)] = validation_data[needed_columns]

                                # scaling features and target based on feature_scaler and target_scaler
                                training_data, validation_data = data_scaling(train_data = training_data, test_data = validation_data, feature_scaler = feature_scaler, target_scaler = target_scaler)
                                
                                # add the current fold data, model name and model parameters to the list of pool function arguments
                                if callable(model):
                                    pool_list.append(tuple((training_data, validation_data, model.__name__, model_type, model_parameters, labels, 0)))
                                else:
                                    pool_list.append(tuple((training_data, validation_data, model, model_type, model_parameters, labels, 0)))



            # running the models in parallel
            parallel_output = prediction_pool.map(partial(parallel_run),tqdm(list(pool_list)))
            prediction_pool.close()
            prediction_pool.join()
        
        ####################### geting outputs of the base models
        
        pool_index = 0 # the index of pool results
        
        for feature_set_number in range(len(feature_sets_indices[history-1])):
            for model_number, model in enumerate(models_list):
                model_name = models_name_list[model_number]
                if model_name in base_models_name_list:
                    for model_parameters_number in range(len(models_parameter_list[model_number])):
                        for fold_number in range(1, fold_total_number + 1):
                            # save the models prediction output for the current fold
                            fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = parallel_output[pool_index][0]
                            fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = parallel_output[pool_index][1]
                            number_of_parameters[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = parallel_output[pool_index][2]
                            
                            pool_index = pool_index + 1
                                                     
        
        ####################### calculate and save the performance
        
        for feature_set_number in range(len(feature_sets_indices[history-1])):
            
            for model_number, model in enumerate(models_list):
                model_name = models_name_list[model_number]
                if model_name in base_models_name_list:
                                                     
                    for model_parameters_number in range(len(models_parameter_list[model_number])):

                        # initializing a dictionary for hold each folds training and validation error for the current model
                        fold_validation_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                        fold_training_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}

                        for fold_number in range(1, fold_total_number + 1):

                            # descale the predictions
                            normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = target_descale(
                                                                                                                scaled_data = fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                                                                                                base_data = list(np.array(target_real_values['training'][(fold_number, history)]['Target']).reshape(-1)), 
                                                                                                                scaler = target_scaler)
                            normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = target_descale(
                                                                                                                scaled_data = fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)], 
                                                                                                                base_data = list(np.array(target_real_values['training'][(fold_number, history)]['Target']).reshape(-1)), 
                                                                                                                scaler = target_scaler)

                            # get the normal values of the target variable and predictions for the cumulative, differential,
                            # and moving average modes
                            
                            normal_target_real_values['training'][(fold_number, history)], normal_target_real_values['validation'][(fold_number, history)],\
                            normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],\
                            normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = \
                                                    get_normal_target(training_target = target_real_values['training'][(fold_number, history)].copy(),
                                                                       test_target = target_real_values['validation'][(fold_number, history)].copy(),
                                                                       training_prediction = normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       test_prediction = normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       target_mode = target_mode[history-1], target_granularity = target_granularity[history-1],
                                                                       same_train_test = same_train_validation_sets)

                            performance_target_real_values['training'][(fold_number, history)], performance_target_real_values['validation'][(fold_number, history)],\
                            performance_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],\
                            performance_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = \
                                                                    apply_performance_mode(training_target = normal_target_real_values['training'][(fold_number, history)].copy(),
                                                                       test_target = normal_target_real_values['validation'][(fold_number, history)].copy(),
                                                                       training_prediction = normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       test_prediction = normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       performance_mode = performance_mode, same_train_test = same_train_validation_sets)


                            # calculate and store the performance measure for the current fold
                            for measure in performance_measures:

                                train_data_deficient_flag = 0
                                validation_data_deficient_flag = 0
                                train_df = performance_target_real_values['training'][(fold_number, history)]
                                validation_df = performance_target_real_values['validation'][(fold_number, history)]


                                if measure != 'MASE':
                                    train_true_values = list(np.array(train_df['Normal target']).reshape(-1))
                                    train_predicted_values = performance_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)]
                                    train_trivial_values = []
                                    validation_true_values = list(np.array(validation_df['Normal target']).reshape(-1))
                                    validation_predicted_values = performance_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)]
                                    validation_trivial_values = []

                                if measure == 'MASE':
                                    train_true_values, train_predicted_values, train_trivial_values,\
                                    validation_true_values, validation_predicted_values,\
                                    validation_trivial_values = get_trivial_values(
                                    train_true_values_df = train_df, validation_true_values_df = validation_df,
                                        train_prediction = performance_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                        validation_prediction = performance_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)], 
                                        forecast_horizon = forecast_horizon, granularity = granularity[history-1], same_train_validation = same_train_validation_sets)

                                    if len(train_trivial_values)<1:
                                        train_data_deficient_flag = 1
                                    if len(validation_trivial_values)<1:
                                        validation_data_deficient_flag = 1
                                        if performance_benchmark == 'MASE':
                                            raise Exception("There are not enough temporal units in the data to measure 'MASE'.")


                                if validation_data_deficient_flag ==0:
                                    fold_validation_error[fold_number][measure] = performance(true_values = validation_true_values,
                                                                                              predicted_values = validation_predicted_values, 
                                                                                              performance_measures = list([measure]), 
                                                                                              trivial_values = validation_trivial_values,
                                                                                              model_type = model_type,
                                                                                              num_params = number_of_parameters[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                                                                              labels = labels)
                                else:
                                    fold_validation_error[fold_number][measure] = [None]

                                if train_data_deficient_flag ==0:
                                    fold_training_error[fold_number][measure] = performance(true_values = train_true_values,
                                                                                              predicted_values = train_predicted_values,
                                                                                              performance_measures = list([measure]),
                                                                                              trivial_values = train_trivial_values,
                                                                                              model_type = model_type,
                                                                                              num_params = number_of_parameters[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                                                                              labels = labels)
                                else:
                                    fold_training_error[fold_number][measure] = [None]

                        # calculating and storing the cross-validation final performance measure by taking the average of the folds performance measure
                        for measure in performance_measures:
                            fold_validation_error_list = list([fold_validation_error[fold_number][measure][0] for fold_number in range(1, fold_total_number + 1)])
                            fold_training_error_list = list([fold_training_error[fold_number][measure][0] for fold_number in range(1, fold_total_number + 1)])


                            if not None in fold_validation_error_list:
                                validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = np.mean(fold_validation_error_list)
                            else:
                                validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = None
                            if not None in fold_training_error_list:
                                training_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = np.mean(fold_training_error_list)
                            else:
                                training_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = None


                            # update the best history length and best feature set based on the value of performance_benchmark measure
                            if measure == performance_benchmark:
                                if measure in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] < models_min_validation_error[model_name]:
                                        models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]
                                        models_best_history_length[model_name] = history
                                        models_best_feature_set_number[model_name] = feature_set_number 
                                    # update the best model_parameters for the current history and featureset
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] < model_parameters_min_validation_error[model_name][(history, feature_set_number)]:
                                        models_best_parameters_number[model_name][(history, feature_set_number)] = model_parameters_number
                                        model_parameters_min_validation_error[model_name][(history, feature_set_number)] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]
                                else:
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] > models_min_validation_error[model_name]:
                                        models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]
                                        models_best_history_length[model_name] = history
                                        models_best_feature_set_number[model_name] = feature_set_number
                                    # update the best model_parameters for the current history and featureset
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] > model_parameters_min_validation_error[model_name][(history, feature_set_number)]:
                                        models_best_parameters_number[model_name][(history, feature_set_number)] = model_parameters_number
                                        model_parameters_min_validation_error[model_name][(history, feature_set_number)] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]

        ####################### Running mixed models
        
        if len(mixed_models_name_list)>0:
            
            with get_context("spawn").Pool(processes = Number_of_cpu) as prediction_pool:
            
                # initializing the pool for parallel run
                pool_list = [] # list of all the different combination of the arguments of pool function
                print('\nMixed models are running.\n')

                for feature_set_number in range(len(feature_sets_indices[history-1])):

                    for model_number, model in enumerate(models_list):
                        model_name = models_name_list[model_number]

                        if model_name in mixed_models_name_list:

                            for model_parameters_number in range(len(models_parameter_list[model_number])):
                                model_parameters = models_parameter_list[model_number][model_parameters_number]

                                for fold_number in range(1, fold_total_number + 1):

                                    training_data = target_real_values['training'][(fold_number, history)].copy()
                                    validation_data = target_real_values['validation'][(fold_number, history)].copy()

                                    for base_model_name in base_models_name_list:

                                        base_model_parameters_number = models_best_parameters_number[base_model_name][(history, feature_set_number)]
                                        base_train_predictions = fold_training_predictions[base_model_name][(fold_number, history, feature_set_number, base_model_parameters_number)]
                                        base_validation_predictions = fold_validation_predictions[base_model_name][(fold_number, history, feature_set_number, base_model_parameters_number)]


                                        if model_type == 'classification':
                                            total_class_number = 1 if base_train_predictions.shape[1] == 2 else base_train_predictions.shape[1]
                                            for class_num in range(total_class_number):
                                                training_data.loc[:,(base_model_name+str(class_num))] = list(base_train_predictions[:,class_num])
                                                validation_data.loc[:,(base_model_name+str(class_num))] = list(base_validation_predictions[:,class_num])
                                        else:
                                            training_data.loc[:,(base_model_name)] = base_train_predictions
                                            validation_data.loc[:,(base_model_name)] = base_validation_predictions


                                    if model_parameters is not None:
                                        if 'n_neighbors' in model_parameters.keys():
                                            if type(model_parameters['n_neighbors']) == int:
                                                if (model_parameters['n_neighbors']>len(training_data)) and (knn_alert_flag == 0):
                                                    print("\nWarning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
                                                    print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]\n")
                                                    knn_alert_flag = 1

                                    # scaling features and target based on feature_scaler and target_scaler
                                    training_data, validation_data = data_scaling(train_data = training_data, test_data = validation_data, feature_scaler = None, target_scaler = target_scaler)

                                    # add the current fold data, model name and model parameters to the list of pool function arguments
                                    if callable(model):
                                        pool_list.append(tuple((training_data, validation_data, model.__name__, model_type, model_parameters, labels, 0)))
                                    else:
                                        pool_list.append(tuple((training_data, validation_data, model, model_type, model_parameters, labels, 0)))

                # running the models in parallel
                parallel_output = prediction_pool.map(partial(parallel_run),tqdm(list(pool_list)))
                prediction_pool.close()
                prediction_pool.join()
            
            
        ####################### geting the outputs of mixed models
        
        pool_index = 0 # the index of pool results
        
        for feature_set_number in range(len(feature_sets_indices[history-1])):
            for model_number, model in enumerate(models_list):
                model_name = models_name_list[model_number]
                if model_name in mixed_models_name_list:
                    for model_parameters_number in range(len(models_parameter_list[model_number])):
                        for fold_number in range(1, fold_total_number + 1):
                            # save the models prediction output for the current fold
                            fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = parallel_output[pool_index][0]
                            fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = parallel_output[pool_index][1]
                            number_of_parameters[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = parallel_output[pool_index][2]

                            pool_index = pool_index + 1
        
        ####################### calculate and save the performance
        
        for feature_set_number in range(len(feature_sets_indices[history-1])):
            
            for model_number, model in enumerate(models_list):
                model_name = models_name_list[model_number]
                if model_name in mixed_models_name_list:
                                                     
                    for model_parameters_number in range(len(models_parameter_list[model_number])):

                        # initializing a dictionary for hold each folds training and validation error for the current model
                        fold_validation_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                        fold_training_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}

                        for fold_number in range(1, fold_total_number + 1):

                            # descale the predictions
                            normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = target_descale(
                                                                                                                scaled_data = fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                                                                                                base_data = list(np.array(target_real_values['training'][(fold_number, history)]['Target']).reshape(-1)), 
                                                                                                                scaler = target_scaler)
                            normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = target_descale(
                                                                                                                scaled_data = fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)], 
                                                                                                                base_data = list(np.array(target_real_values['training'][(fold_number, history)]['Target']).reshape(-1)), 
                                                                                                                scaler = target_scaler)

                            # get the normal values of the target variable and predictions for the cumulative, differential,
                            # and moving average modes

                            normal_target_real_values['training'][(fold_number, history)], normal_target_real_values['validation'][(fold_number, history)],\
                            normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],\
                            normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = \
                                                    get_normal_target(training_target = target_real_values['training'][(fold_number, history)].copy(),
                                                                       test_target = target_real_values['validation'][(fold_number, history)].copy(),
                                                                       training_prediction = normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       test_prediction = normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       target_mode = target_mode[history-1], target_granularity = target_granularity[history-1],
                                                                       same_train_test = same_train_validation_sets)

                            performance_target_real_values['training'][(fold_number, history)], performance_target_real_values['validation'][(fold_number, history)],\
                            performance_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],\
                            performance_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)] = \
                                                                    apply_performance_mode(training_target = normal_target_real_values['training'][(fold_number, history)].copy(),
                                                                       test_target = normal_target_real_values['validation'][(fold_number, history)].copy(),
                                                                       training_prediction = normal_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       test_prediction = normal_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)].copy(),
                                                                       performance_mode = performance_mode, same_train_test = same_train_validation_sets)


                            # calculate and store the performance measure for the current fold
                            for measure in performance_measures:

                                train_data_deficient_flag = 0
                                validation_data_deficient_flag = 0
                                train_df = performance_target_real_values['training'][(fold_number, history)]
                                validation_df = performance_target_real_values['validation'][(fold_number, history)]


                                if measure != 'MASE':
                                    train_true_values = list(np.array(train_df['Normal target']).reshape(-1))
                                    train_predicted_values = performance_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)]
                                    train_trivial_values = []
                                    validation_true_values = list(np.array(validation_df['Normal target']).reshape(-1))
                                    validation_predicted_values = performance_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)]
                                    validation_trivial_values = []

                                if measure == 'MASE':
                                    train_true_values, train_predicted_values, train_trivial_values,\
                                    validation_true_values, validation_predicted_values,\
                                    validation_trivial_values = get_trivial_values(
                                    train_true_values_df = train_df, validation_true_values_df = validation_df,
                                        train_prediction = performance_fold_training_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                        validation_prediction = performance_fold_validation_predictions[model_name][(fold_number, history, feature_set_number, model_parameters_number)], 
                                        forecast_horizon = forecast_horizon, granularity = granularity[history-1], same_train_validation = same_train_validation_sets)

                                    if len(train_trivial_values)<1:
                                        train_data_deficient_flag = 1
                                    if len(validation_trivial_values)<1:
                                        validation_data_deficient_flag = 1
                                        if performance_benchmark == 'MASE':
                                            raise Exception("There are not enough temporal units in the data to measure 'MASE'.")


                                if validation_data_deficient_flag ==0:
                                    fold_validation_error[fold_number][measure] = performance(true_values = validation_true_values,
                                                                                              predicted_values = validation_predicted_values, 
                                                                                              performance_measures = list([measure]), 
                                                                                              trivial_values = validation_trivial_values,
                                                                                              model_type = model_type,
                                                                                              num_params = number_of_parameters[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                                                                              labels = labels)
                                else:
                                    fold_validation_error[fold_number][measure] = [None]

                                if train_data_deficient_flag ==0:
                                    fold_training_error[fold_number][measure] = performance(true_values = train_true_values,
                                                                                              predicted_values = train_predicted_values,
                                                                                              performance_measures = list([measure]),
                                                                                              trivial_values = train_trivial_values,
                                                                                              model_type = model_type,
                                                                                              num_params = number_of_parameters[model_name][(fold_number, history, feature_set_number, model_parameters_number)],
                                                                                              labels = labels)
                                else:
                                    fold_training_error[fold_number][measure] = [None]

                        # calculating and storing the cross-validation final performance measure by taking the average of the folds performance measure
                        for measure in performance_measures:
                            fold_validation_error_list = list([fold_validation_error[fold_number][measure][0] for fold_number in range(1, fold_total_number + 1)])
                            fold_training_error_list = list([fold_training_error[fold_number][measure][0] for fold_number in range(1, fold_total_number + 1)])


                            if not None in fold_validation_error_list:
                                validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = np.mean(fold_validation_error_list)
                            else:
                                validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = None
                            if not None in fold_training_error_list:
                                training_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = np.mean(fold_training_error_list)
                            else:
                                training_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] = None


                            # update the best history length and best feature set based on the value of performance_benchmark measure
                            if measure == performance_benchmark:
                                if measure in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] < models_min_validation_error[model_name]:
                                        models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]
                                        models_best_history_length[model_name] = history
                                        models_best_feature_set_number[model_name] = feature_set_number 
                                    # update the best model_parameters for the current history and featureset
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] < model_parameters_min_validation_error[model_name][(history, feature_set_number)]:
                                        models_best_parameters_number[model_name][(history, feature_set_number)] = model_parameters_number
                                        model_parameters_min_validation_error[model_name][(history, feature_set_number)] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]
                                else:
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] > models_min_validation_error[model_name]:
                                        models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]
                                        models_best_history_length[model_name] = history
                                        models_best_feature_set_number[model_name] = feature_set_number
                                    # update the best model_parameters for the current history and featureset
                                    if validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)] > model_parameters_min_validation_error[model_name][(history, feature_set_number)]:
                                        models_best_parameters_number[model_name][(history, feature_set_number)] = model_parameters_number
                                        model_parameters_min_validation_error[model_name][(history, feature_set_number)] = validation_errors[measure][model_name][(history, feature_set_number, model_parameters_number)]

        
        # save the ranked_features for each history if feature_selection_type is 
        # feature and for the all histories othervise
        if feature_selection_type == 'feature':
            ranked_features_to_save.append([item for item in list(train_data.columns) if item not in ['spatial id', 'temporal id', 'Target', 'Normal target']])
        elif history == max_history:
            ranked_features_to_save = [item for item in list(train_data.columns) if item not in ['spatial id', 'temporal id', 'Target', 'Normal target']]
            
            
    #################################################### saving predictions
    
    # save the real and predicted value of target variable in training and validation set for each model
    
    if (save_predictions == True) and (fold_total_number == 1): # if cross validation mode is on, predictions are not saved
        
        save_prediction_data_frame(models_name_list = models_name_list, target_real_values = normal_target_real_values, 
                                   fold_validation_predictions = normal_fold_validation_predictions, fold_training_predictions = normal_fold_training_predictions,
                                   models_best_history_length = models_best_history_length, models_best_feature_set_number = models_best_feature_set_number,
                                   models_best_parameters_number = models_best_parameters_number, forecast_horizon = forecast_horizon, 
                                   data_temporal_size = number_of_temporal_units, prediction_type = 'training', model_type = model_type, labels = labels)
        if same_train_validation_sets == False:
            save_prediction_data_frame(models_name_list = models_name_list, target_real_values = normal_target_real_values, 
                                   fold_validation_predictions = normal_fold_validation_predictions, fold_training_predictions = normal_fold_training_predictions,
                                   models_best_history_length = models_best_history_length, models_best_feature_set_number = models_best_feature_set_number,
                                   models_best_parameters_number = models_best_parameters_number, forecast_horizon = forecast_horizon, 
                                   data_temporal_size = number_of_temporal_units, prediction_type = 'validation', model_type = model_type, labels = labels)
                                                         
    #################################################### reporting performance
    
    if performance_report == True:
        if same_train_validation_sets == False:
            report_performance(errors_dict = validation_errors, max_history = max_history, ordered_covariates_or_features = ordered_covariates_or_features,
                              feature_sets_indices = feature_sets_indices, performance_measures = performance_measures, feature_selection_type = feature_selection_type,
                              models_name_list = models_name_list, models_best_parameters_number = models_best_parameters_number, forecast_horizon = forecast_horizon,
                              data_temporal_size = number_of_temporal_units, report_type = 'validation')
        report_performance(errors_dict = training_errors, max_history = max_history, ordered_covariates_or_features = ordered_covariates_or_features,
                          feature_sets_indices = feature_sets_indices, performance_measures = performance_measures, feature_selection_type = feature_selection_type,
                          models_name_list = models_name_list, models_best_parameters_number = models_best_parameters_number, forecast_horizon = forecast_horizon,
                          data_temporal_size = number_of_temporal_units, report_type = 'training')
        
    #################################################### save the ranking
    
    if save_ranked_features == True:
        save_ranking(ranked_features_to_save, feature_selection_type, ordered_covariates_or_features)
    
    #################################################### finding best model and overall best history length and feature set
    
    for model_number, model_name in enumerate(models_name_list):
        if performance_benchmark in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
            if models_min_validation_error[model_name] < overall_min_validation_error:
                overall_min_validation_error = models_min_validation_error[model_name]
                best_history_length = models_best_history_length[model_name]
                best_feature_set_number = models_best_feature_set_number[model_name]
                best_feature_sets_indices = feature_sets_indices[best_history_length-1][best_feature_set_number]
                best_parameters_number = models_best_parameters_number[model_name][(best_history_length, best_feature_set_number)]             
                best_parameters = models_parameter_list[model_number][best_parameters_number]
                best_model = models_list[model_number]
                best_model_number = model_number
                best_model_name = model_name
                
        else:
            if models_min_validation_error[model_name] > overall_min_validation_error:
                overall_min_validation_error = models_min_validation_error[model_name]
                best_history_length = models_best_history_length[model_name]
                best_feature_set_number = models_best_feature_set_number[model_name]
                best_feature_sets_indices = feature_sets_indices[best_history_length-1][best_feature_set_number]
                best_parameters_number = models_best_parameters_number[model_name][(best_history_length, best_feature_set_number)]              
                best_parameters = models_parameter_list[model_number][best_parameters_number]
                best_model = models_list[model_number]
                best_model_number = model_number
                best_model_name = model_name
    
    
    # determining the base models of the best model (if it's a mixed model) with the selected parameters
    # in the grid search
    if best_model_name in mixed_models_name_list:
        best_model_base_models = []
        for model_number, model in enumerate(models_list):
            model_name = models_name_list[model_number]
            if model_name in base_models_name_list:
                base_model_best_parameters_number = models_best_parameters_number[model_name][(best_history_length, best_feature_set_number)]              
                base_model_best_parameters = models_parameter_list[model_number][base_model_best_parameters_number]
                if base_model_best_parameters is not None:
                    best_model_base_models.append({model:base_model_best_parameters})
                else:
                    best_model_base_models.append(model)
    else:
        best_model_base_models = None
    
    # training the best model using the data with the overall best history length and feature set

    data = data_list[best_history_length-1].copy()
    
    # separating the test part
    raw_train_data, _ , raw_testing_data , _ = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                              instance_validation_size = None, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                              instance_random_partitioning = instance_random_partitioning, granularity = granularity[best_history_length-1], verbose = 0)
    
    best_train_data = raw_train_data
    best_feature_or_covariate_set = [ordered_covariates_or_features[best_history_length-1][index] for index in best_feature_sets_indices]
    # select the features
    best_train_data = select_features(data = best_train_data.copy(), ordered_covariates_or_features = best_feature_or_covariate_set)
    
    best_train_data, _ = data_scaling(train_data = best_train_data, test_data = best_train_data, feature_scaler = feature_scaler, target_scaler = target_scaler)
    
    _, _, best_trained_model = train_evaluate(training_data = best_train_data,
                                              validation_data = best_train_data,
                                              model = best_model, model_type = model_type,
                                              model_parameters = best_parameters,
                                              labels = labels,
                                              base_models = best_model_base_models,
                                              verbose = 0)
        
    
    path_to_user_defined_models_dumb_files = imp.find_module('stpredict')[1]+"/user_defined_models.py"
    if os.path.isfile(path_to_user_defined_models_dumb_files):
        os.remove(path_to_user_defined_models_dumb_files)
    
    return best_model, best_parameters, best_history_length, best_feature_or_covariate_set, best_model_base_models, best_trained_model
