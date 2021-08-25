import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import os
    import shutil
    import sys

    import pandas as pd
    import random

    from .configurations import *
    from .get_future_data import get_future_data
    from .get_target_quantities import get_target_quantities
    from .train_validate import train_validate
    from .train_test import train_test
    from .predict_future import predict_future
    from .plot_prediction import plot_prediction
    from .get_target_temporal_ids import get_target_temporal_ids
    from .performance_summary import performance_summary
    from .performance_summary import performance_bar_plot


def predict(data: list,
            forecast_horizon: int = 1,
            feature_sets: dict = {'covariate': 'mRMR'},
            forced_covariates: list = [],
            models: list = ['knn'],
            mixed_models: list = [],
            model_type: str = 'regression',
            splitting_type: str = 'training-validation',
            instance_testing_size: int or float = 0.2,
            instance_validation_size: int or float = 0.3,
            instance_random_partitioning: bool = False,
            fold_total_number: int = 5,
            feature_scaler: str = None,
            target_scaler: str = None,
            performance_benchmark: str = 'MAPE',
            performance_measures: list = ['MAPE'],
            performance_mode: str = 'normal',
            scenario: str or None = 'current',
            validation_performance_report: bool = True,
            testing_performance_report: bool = True,
            save_predictions: bool = True,
            plot_predictions: bool = False,
            verbose: int = 0):
    
    
    """
    Args:
        data:
        forecast_horizon:
        feature_sets:
        forced_covariates:
        models:
        mixed_models:
        model_type:
        splitting_type:
        instance_testing_size:
        instance_validation_size:
        instance_random_partitioning:
        fold_total_number:
        feature_scaler:
        target_scaler:
        performance_benchmark:
        performance_measures:
        performance_mode:
        scenario:
        validation_performance_report:
        testing_performance_report:
        save_predictions:
        plot_predictions:
        verbose:
    Returns:
    """
    # input checking
    # data
    if not isinstance(data, list):
        sys.exit("The input 'data' must be a list of DataFrames or a list of data addresses.")
    str_check = [isinstance(d, str) for d in data]
    df_check = [isinstance(d, pd.DataFrame) for d in data]
    if not (all(str_check) or all(df_check)):
        sys.exit("The input 'data' must be a list of DataFrames or a list data addresses.")
    # forecast_horizon
    if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
        sys.exit("The input 'forecast_horizon' must be integer and greater than or equal to one.")
    # feature_scaler
    if feature_scaler not in FEATURE_SCALERS:
        sys.exit(f"The input 'feature_scaler' must be string and one of the following options:\n"
                 f"{FEATURE_SCALERS}")
    # target_scaler
    if target_scaler not in TARGET_SCALERS:
        sys.exit(f"The input 'target_scaler' must be string and one of the following options:\n"
                 f"{TARGET_SCALERS}")
    # feature_sets input checking
    if not (isinstance(feature_sets, dict) and len(feature_sets.keys()) == 1):
        sys.exit("feature_sets input format is not valid.")
    if not (list(feature_sets.keys())[0] in FEATURE_SELECTION_TYPES
            and list(feature_sets.values())[0] in RANKING_METHODS):
        sys.exit("feature_sets input is not valid.")
    # forced_covariates checking
    if not isinstance(forced_covariates, list):
        sys.exit("Error: The input 'forced_covariates' must be a list of covariates or an empty list.")
    # model_type input checking
    if model_type not in MODEL_TYPES:
        sys.exit("model_type input is not valid.")
    models_list = []
    # models input checking
    if not isinstance(models, list):
        sys.exit("models input format is not valid.")
    for model in models:
        if isinstance(model, str):
            if model not in PRE_DEFINED_MODELS:
                sys.exit("models input is not valid.")
            elif model not in models_list:
                models_list.append(model)
            else:models.remove(model)
        elif isinstance(model, dict):
            if len(list(model.keys())) == 1:
                if list(model.keys())[0] not in PRE_DEFINED_MODELS:
                    sys.exit("models input is not valid.")
                elif list(model.keys())[0] not in models_list:
                    models_list.append(list(model.keys())[0])
                else:models.remove(model)
            else:
                sys.exit("models input is not valid.")
        elif callable(model):
            if model.__name__ not in models_list:
                models_list.append(model.__name__)
            else:models.remove(model)
        else:
            sys.exit("Models input is not valid.")
    # mixed_models input checking
    if not isinstance(mixed_models, list):
        sys.exit("Mixed_models input format is not valid.")
    for model in mixed_models:
        if isinstance(model, str):
            if model not in PRE_DEFINED_MODELS:
                sys.exit("Mixed_models input is not valid.")
            elif 'mixed_'+model not in models_list:
                models_list.append('mixed_'+model)
            else:mixed_models.remove(model)
        elif isinstance(model, dict):
            if len(list(model.keys())) == 1:
                if list(model.keys())[0] not in PRE_DEFINED_MODELS:
                    sys.exit("Mixed_models input is not valid.")
                elif 'mixed_'+list(model.keys())[0] not in models_list:
                    models_list.append('mixed_'+list(model.keys())[0])
                else:mixed_models.remove(model)
            else:
                sys.exit("Mixed_models input is not valid.")
        elif callable(model):
            if model.__name__ not in models_list:
                models_list.append(model.__name__)
            else:mixed_models.remove(model)
        else:
            sys.exit("Mixed_models input is not valid.")
    # instance_testing_size input checking
    if not ((isinstance(instance_testing_size, float) and 0 < instance_testing_size < 1) or (
            isinstance(instance_testing_size, int) and instance_testing_size > 0)):
        sys.exit("instance_testing_size input is not valid.")
    # splitting_type input checking
    if splitting_type not in SPLITTING_TYPES:
        sys.exit("splitting_type input is not valid.")
    # instance_validation_size input checking
    if not ((isinstance(instance_validation_size, float) and 0 < instance_validation_size < 1) or (
            isinstance(instance_validation_size, int) and instance_validation_size > 0)):
        sys.exit("instance_validation_size input is not valid.")
    # instance_random_partitioning input checking
    if not isinstance(instance_random_partitioning, bool):
        sys.exit("instance_random_partitioning input is not valid.")
    # fold_total_number input checking
    if not (isinstance(fold_total_number, int), fold_total_number > 1):
        sys.exit("fold_total_number input is not valid.")
    # performance_benchmark input checking
    if performance_benchmark not in PERFORMANCE_BENCHMARKS:
        sys.exit("performance_benchmark input is not valid.")
    # performance_mode input checking
    if not isinstance(performance_mode, str):
        sys.exit("performance_mode input format is not valid.")
    if not any(performance_mode.startswith(performance_mode_starts_with)
               for performance_mode_starts_with in PERFORMANCE_MODES_STARTS_WITH):
        sys.exit("performance_mode input is not valid.")
    # performance_measures input checking
    if not (isinstance(performance_measures, list) and len(performance_measures) > 0):
        sys.exit("performance_measures input format is not valid.")
    for performance_measure in performance_measures:
        if performance_measure not in PERFORMANCE_MEASURES:
            sys.exit("performance_measures input is not valid.")
    # scenario
    if not ((isinstance(scenario, str) and scenario in SCENARIOS) or scenario is None):
        sys.exit("scenario input is not valid.")
    # validation_performance_report input checking
    if not isinstance(validation_performance_report, bool):
        sys.exit("validation_performance_report input is not valid.")
    # testing_performance_report input checking
    if not isinstance(testing_performance_report, bool):
        sys.exit("testing_performance_report input is not valid.")
    # save_predictions input checking
    if not isinstance(save_predictions, bool):
        sys.exit("save_predictions input is not valid.")
    # plot_predictions input checking
    if not isinstance(plot_predictions, bool):
        sys.exit("plot_predictions input is not valid.")
    elif (plot_predictions == True) and (save_predictions == False):
        sys.exit("For plotting the predictions, both plot_predictions and save_predictions inputs must be set to TRUE.")
    elif (plot_predictions == True) and (model_type == 'classification'):
        sys.exit("The plot_predictions input can be set to True only for regression model_type.")
        
    # verbose input checking
    if verbose not in VERBOSE_OPTIONS:
        sys.exit("verbose input is not valid.")

    # removing prediction and performance directories and test_process_backup csv file
    if os.path.exists('prediction'):
        shutil.rmtree('prediction')
    if os.path.exists('performance'):
        shutil.rmtree('performance')
    if os.path.isfile('test_process_backup.csv'):
        os.remove('test_process_backup.csv')

    # data preparing
    if isinstance(data[0], str):
        try:
            data = [pd.read_csv(d).sort_values(by=['temporal id', 'spatial id']) for d in data]
        except Exception as e:
            sys.exit(str(e))

    # forced_covariates manipulation
    forced_covariates = list(set(forced_covariates))
    forced_covariates = [forced_covariate
                         for forced_covariate in forced_covariates
                         if forced_covariate is not None and forced_covariate != '']

    # classification checking
    labels = None
    if model_type == 'classification':
        if not set(performance_measures) <= set(CLASSIFICATION_PERFORMANCE_MEASURES):
            sys.exit("Error: The input 'performance_measures' is not valid according to 'model_type=classification'.")
        if performance_benchmark not in CLASSIFICATION_PERFORMANCE_BENCHMARKS:
            sys.exit("Error: The input 'performance_benchmark' is not valid according to 'model_type=classification'.")
        if performance_mode != 'normal':
            performance_mode = 'normal'
            print("Warning: The input 'performance_mode' is set to 'normal' according to model_type=classification'.")
        if target_scaler is not None:
            target_scaler = None
            print("Warning: The input 'target_scaler' is set to None according to model_type=classification'.")
        target_column_name = list(filter(lambda x: x.startswith('Target'), data[0].columns.values))[0]
        labels = data[0].loc[:, target_column_name].unique().tolist()
        labels = [label for label in labels if not (label is None or str(label) == 'nan')]
        if len(labels) < 2:
            sys.exit("Error: The labels length must be at least two.")
            
    # set test_type
    test_type = 'whole-as-one'
    
    # get target quantities
    granularity = [1]*len(data)
    for index in range(len(data)):
        target_mode, target_granularity, granularity[index], _ = get_target_quantities(data=data[index].copy())
        data[index], _ = get_target_temporal_ids(temporal_data = data[index].copy(), forecast_horizon = forecast_horizon,
                                              granularity = granularity[index])
        if model_type == 'classification':
            if not target_mode == 'normal':
                sys.exit(
                    "Error: The parameter 'target_mode' must be 'normal' according to 'model_type=classification'.")
            if not target_granularity == 1:
                sys.exit(
                    "Error: The parameter 'target_mode' must be 'normal' according to 'model_type=classification'.")
            if not granularity[index] == 1:
                sys.exit(
                    "Error: The temporal scale of input data must not be transformed according to 'model_type=classification'.")
    
    data, future_data = get_future_data(data=[d.copy() for d in data],
                                        forecast_horizon=forecast_horizon)
    
    # change the name of temporal id to be identified as shifted to target time point
    for index in range(len(data)):
        data[index] = data[index].rename(columns = {'temporal id':'target temporal id'})
        future_data[index] = future_data[index].rename(columns = {'temporal id':'target temporal id'})
    

    # main process
    data_temporal_ids = [d['target temporal id'].unique() for d in data]
    # train_validate
    if verbose > 0:
        print(100 * '-')
        print('Train Validate Process')
    best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, base_models, _ = \
        train_validate(data=[d.copy() for d in data],
                       forecast_horizon=forecast_horizon,
                       feature_scaler=feature_scaler,
                       target_scaler=target_scaler,
                       feature_sets=feature_sets,
                       forced_covariates=forced_covariates,
                       model_type=model_type,
                       labels=labels,
                       models=models,
                       mixed_models=mixed_models,
                       instance_testing_size=instance_testing_size,
                       splitting_type=splitting_type,
                       instance_validation_size=instance_validation_size,
                       instance_random_partitioning=instance_random_partitioning,
                       fold_total_number=fold_total_number,
                       performance_benchmark=performance_benchmark,
                       performance_measures=performance_measures,
                       performance_report=validation_performance_report,
                       save_predictions=save_predictions,
                       verbose=verbose)

    # train_test
    if verbose > 0:
        print(100 * '-')
        print('Train Test Process')
    test_trained_model = train_test(data=data[best_history_length - 1].copy(),
                                                   forecast_horizon=forecast_horizon,
                                                   history_length=best_history_length,
                                                   feature_scaler=feature_scaler,
                                                   target_scaler=target_scaler,
                                                   feature_or_covariate_set=best_feature_or_covariate_set,
                                                   model_type=model_type,
                                                   labels=labels,
                                                   model=best_model,
                                                   base_models = base_models,
                                                   model_parameters=best_model_parameters,
                                                   instance_testing_size=instance_testing_size,
                                                   performance_measures=performance_measures,
                                                   performance_mode=performance_mode,
                                                   performance_report=testing_performance_report,
                                                   save_predictions=save_predictions,
                                                   verbose=verbose)
    # predict_future
    if verbose > 0:
        print(100 * '-')
        print('Forecast Training process\n')
    best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, base_models, _ = \
        train_validate(data=[d[d['target temporal id'].isin((
                            data_temporal_ids[index][:] if (forecast_horizon*granularity[index])-1 == 0 else data_temporal_ids[index][:-((forecast_horizon*granularity[index])-1)]))].copy()
                            for index, d in enumerate(data)],
                       forecast_horizon=forecast_horizon,
                       feature_scaler=feature_scaler,
                       target_scaler=target_scaler,
                       feature_sets=feature_sets,
                       forced_covariates=forced_covariates,
                       model_type=model_type,
                       labels=labels,
                       models=models,
                       mixed_models=mixed_models,
                       instance_testing_size=0,
                       splitting_type=splitting_type,
                       instance_validation_size=instance_validation_size,
                       instance_random_partitioning=instance_random_partitioning,
                       fold_total_number=fold_total_number,
                       performance_benchmark=performance_benchmark,
                       performance_measures=performance_measures,
                       performance_report=False,
                       save_predictions=False,
                       verbose=0)

    if verbose > 0:
        print(100 * '-')
        print('Predict Future Process')
    best_data = data[best_history_length - 1].copy()
    best_future_data = future_data[best_history_length - 1].copy()
    best_data_temporal_ids = best_data['target temporal id'].unique()
    temp = forecast_horizon*granularity[best_history_length - 1] - 1
    trained_model = predict_future(data=best_data[best_data['target temporal id'].isin((best_data_temporal_ids
                                                                                 if temp == 0
                                                                                 else best_data_temporal_ids[:-temp]
                                                                                 ))].copy(),
                                   future_data=best_future_data.copy(),
                                   forecast_horizon=forecast_horizon,
                                   feature_scaler=feature_scaler,
                                   target_scaler=target_scaler,
                                   feature_or_covariate_set=best_feature_or_covariate_set,
                                   model_type=model_type,
                                   labels=labels,
                                   model=best_model,
                                   base_models = base_models,
                                   model_parameters=best_model_parameters,
                                   scenario=scenario,
                                   save_predictions=save_predictions,
                                   verbose=verbose)
            
    if (validation_performance_report == True and testing_performance_report == True):
        performance_bar_plot(forecast_horizon,test_type,performance_benchmark)
        performance_summary(forecast_horizon,test_type,performance_benchmark)
        
    if plot_predictions == True:
        if len(data[0]['spatial id'].unique())<3:
            spatial_ids = data[0]['spatial id'].unique()
        else:
            spatial_ids = list(random.sample(list(data[0]['spatial id'].unique()),3))
        plot_prediction(data = data[0].copy(), test_type = test_type, forecast_horizon = forecast_horizon,
                         plot_type = 'test', granularity = granularity[0], spatial_ids = spatial_ids)
        plot_prediction(data = data[0].copy(), test_type = test_type, forecast_horizon = forecast_horizon,
                         plot_type = 'future', granularity = granularity[0], spatial_ids = spatial_ids)


    return None


if __name__ == '__main__':
    input_data = [f'usa_data/historical_data h={i}.csv' for i in range(1, 4)]
    # input_data = [f'canada_data/historical_data h={i}.csv' for i in range(1, 6)]
    # input_data = [f'my_data/historical_data h={i}.csv' for i in range(1, 2)]
    # input_data = [f'mbp_data/historical_data h={i}.csv' for i in range(1, 2)]
    predict(data=input_data,
            forecast_horizon=4,
            feature_scaler='logarithmic',
            target_scaler=None,
            feature_sets={'feature': 'mRMR'},
            forced_covariates=[],
            model_type='regression',
            models=['knn', 'gbm'],
            instance_testing_size=0.2,
            splitting_type='training-validation',
            instance_validation_size=0.3,
            instance_random_partitioning=False,
            fold_total_number=5,
            performance_benchmark='MAE',
            performance_mode='normal',
            performance_measures=['MAE', 'MAPE'],
            scenario='current',
            validation_performance_report=True,
            testing_performance_report=True,
            save_predictions=True,
            verbose=0)