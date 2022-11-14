import pandas as pd
from stpredict.preprocess import data_preprocess
from stpredict.predict import lafopafo

## Load data from file
data = pd.read_csv('Dataset1_100x100m.csv')

## Specifying the spatial covariates which has unique value for each spatial unit
spatial_covariates = ['Latitude', 'Longitude', 'Elevation', 'Slope',
       'Aspect', 'Northerness', 'Easterness', 'Dist2BorderS']

## Specifying the temporal covariates whose values change over time
temporal_covariates = ['MPB', 'Tmax', 'Tmin_Summer',
       'Tmin_Winter', 'DegreeDays', 'ColdTolerance', 'RelativeHumidity', 'SMI',
       'WindSpeed', 'PeakEmergence', 'PineCover', 'PineHeight', 'PineAge',
       'Stems', 'BP0', 'BP1', 'BP2', 'BP3', 'BP0red', 'BP1red', 'BP2red',
       'BP3red', 'BP0man', 'BP1man', 'BP2man', 'BP3man']


## Specifying the column name of the finest spatial and temporal scale units' IDs, 
## temporal and spatial covariates and the target variable
column_identifier = {'spatial id level 1': 'CellID',
                     'temporal id level 1': 'Year',
                     'temporal covariates': temporal_covariates,
                     'spatial covariates': spatial_covariates,
                     'target': 'MPB'}


## Specifying the number of years ahead that must be forecasted and number of years 
## in the past for each temporal covariate which will be considered in the model (here 
## we consider 5 years of history for all covariates)
forecast_horizon = 2
max_history = 5
history_length = {key : max_history for key in temporal_covariates}

def main():

    ## Preprocessing data and creating 5 historical data frames containing spatial covariates values,
    ## and temporal covariates values with different history lengths ranges from 1 to 5 years, all containing
    ## the target variable values in the next 2 years
    historical_data_list = data_preprocess(data = data.copy(),
                        forecast_horizon = forecast_horizon,
                        history_length = history_length,
                        column_identifier = column_identifier,
                        spatial_scale_table = None,
                        spatial_scale_level = 1,
                        temporal_scale_level = 1,
                        target_mode = 'normal',
                        imputation = False,
                        aggregation_mode = 'mean',
                        augmentation = False,
                        futuristic_covariates = None,
                        future_data_table = None,
                        save_address = './',
                        verbose = 1)

    ## Training the specified models using data with different history length, evaluate the trained
    ## models and find optimal parameters to predict the future, save the predictions and performance
    ## measurements in CSV files
    lafopafo(data = historical_data_list,
                       forecast_horizon = forecast_horizon,
                       feature_scaler = 'normalize',
                       target_scaler = None,
                       feature_sets = {'covariate': 'mRMR'},
                       forced_covariates = [],
                       model_type = 'classification',
                       models = ['knn','glm','gbm,'nn'],
                       mixed_models = ['glm','nn'],
                       instance_testing_size = 2,
                       splitting_type = 'training-validation',
                       instance_random_partitioning = False,
                       performance_benchmark = 'AUC',
                       performance_mode = 'normal',
                       performance_measures = ['AUC','AUPR'],
                       scenario = 'current',
                       validation_performance_report = True,
                       testing_performance_report = True,
                       save_predictions = True, 
                       verbose = 1)
    
if __name__ == "__main__":
    
    main()