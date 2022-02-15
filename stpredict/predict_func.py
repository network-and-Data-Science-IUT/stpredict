import warnings
from .whole_as_one import whole_as_one
from .one_by_one import lafopafo

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

def predict(data: list,
            forecast_horizon: int = 1,
            feature_sets: dict = {'covariate': 'mRMR'},
            forced_covariates: list = [],
            models: list = ['knn'],
            mixed_models: list = [],
            model_type: str = 'regression',
            test_type: str = 'whole-as-one',
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
            save_ranked_features: bool = True,
            plot_predictions: bool = False,
            verbose: int = 0):
    
        
    if test_type == 'whole-as-one':
        
        whole_as_one(data = data,
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
        
        lafopafo(data = data,
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
