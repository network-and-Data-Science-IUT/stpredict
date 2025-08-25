import warnings
import pandas as pd
import pkg_resources
from .whole_as_one import whole_as_one
from .preprocess import preprocess_data
from .one_by_one import lafopafo

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

def load_covid_data():
    stream = pkg_resources.resource_stream(__name__, 'data/USA_COVID_19_data.csv')
    return pd.read_csv(stream)

def load_earthquake_data():
    stream = pkg_resources.resource_stream(__name__, 'data/earthquake_data.csv')
    return pd.read_csv(stream)

def stpredict(data, forecast_horizon,
                    history_length = 1, 
                    column_identifier = None, 
                    feature_sets = {'covariate': 'mRMR'}, 
                    models = ['knn'], 
                    model_type = 'regression',
                    test_type = 'whole-as-one',
                    mixed_models = [], 
                    performance_benchmark = 'MAPE',
                    performance_measures = ['MAPE'], 
                    performance_mode = 'normal', 
                    splitting_type = 'training-validation',
                    instance_testing_size = 0.2, 
                    instance_validation_size = 0.3,
                    instance_random_partitioning = False,
                    fold_total_number = 5, 
                    imputation = True, 
                    target_mode = 'normal',
                    feature_scaler = None,
                    target_scaler = None, 
                    forced_covariates = [], 
                    futuristic_covariates = None, 
                    scenario = 'current', 
                    future_data_table = None,
                    temporal_scale_level = 1, 
                    spatial_scale_level = 1, 
                    spatial_scale_table = None,
                    aggregation_mode = 'mean', 
                    augmentation = False,
                    validation_performance_report = True, 
                    testing_performance_report = True,
                    save_predictions = True, 
                    save_ranked_features = True,
                    plot_predictions = False, 
                    verbose = 0):



    data_list = preprocess_data(data = data, 
                                forecast_horizon = forecast_horizon, 
                                history_length = history_length,
                                column_identifier = column_identifier, 
                                spatial_scale_table = spatial_scale_table,
                                spatial_scale_level = spatial_scale_level, 
                                temporal_scale_level = temporal_scale_level,
                                target_mode = target_mode, 
                                imputation = imputation, 
                                aggregation_mode = aggregation_mode, 
                                augmentation = augmentation, 
                                futuristic_covariates = futuristic_covariates, 
                                future_data_table = future_data_table, 
                                save_address = None, 
                                verbose = verbose)
    
    if not isinstance(data_list,list):
        data_list = [data_list]
        
    if test_type == 'whole-as-one':
        
        whole_as_one(data = data_list,
                    forecast_horizon = forecast_horizon,
                    feature_sets = feature_sets,
                    forced_covariates = forced_covariates,
                    models = models,
                    mixed_models = mixed_models,
                    model_type = model_type,
                    splitting_type = splitting_type,
                    instance_testing_size = instance_testing_size,
                    instance_validation_size = instance_validation_size,
                    instance_random_partitioning = instance_random_partitioning,
                    fold_total_number = fold_total_number,
                    feature_scaler = feature_scaler,
                    target_scaler = target_scaler,
                    performance_benchmark = performance_benchmark,
                    performance_measures = performance_measures,
                    performance_mode = performance_mode,
                    scenario = scenario,
                    validation_performance_report = validation_performance_report,
                    testing_performance_report = testing_performance_report,
                    save_predictions = save_predictions,
                    save_ranked_features = save_ranked_features,
                    plot_predictions = plot_predictions,
                    verbose = verbose)
        
    elif test_type == 'one-by-one':
        
        lafopafo(data = data_list,
                    forecast_horizon = forecast_horizon,
                    feature_sets = feature_sets,
                    forced_covariates = forced_covariates,
                    models = models,
                    mixed_models = mixed_models,
                    model_type = model_type,
                    instance_testing_size = instance_testing_size,
                    instance_validation_size = instance_validation_size,
                    feature_scaler = feature_scaler,
                    target_scaler = target_scaler,
                    performance_benchmark = performance_benchmark,
                    performance_measures = performance_measures,
                    performance_mode = performance_mode,
                    scenario = scenario,
                    validation_performance_report = validation_performance_report,
                    testing_performance_report = testing_performance_report,
                    save_predictions = save_predictions,
                    save_ranked_features = save_ranked_features,
                    plot_predictions = plot_predictions,
                    verbose = verbose)
        
    else:
        raise ValueError("The test_type input must be 'whole-as-one' or 'one-by-one'.")

    return None
