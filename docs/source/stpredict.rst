stpredict
=========

**Description**

Performing all steps of data preprocessing, training, validation and forecasting. This steps will be implemented in two phases. First phase is preprocessing. In this phase first the imputation of missing values will be performed; and in the second step the temporal and spatial scales of data are transformed to the user's desired scale for prediction. Then in the third step, the target variable will be modified based on the user specified mode, and the last step is to reform the data to the historical format containing the historical values of input data covariates and values of the target variable at the forecast horizon. In addition, if the user prefers to use the values of some covariates in the future temporal units for prediction, the name of these covariates could be specified using the futuristic_covariates argument. The second phase is prediction. In this phase firstly the best model, history length, and feature set are selected based on models performance on the validation set and then the selected configuration will be used to predict the target variable values for the test set and report the prediction performance. The predictions of the target variable values for the temporal units in the future which are obtained using the selected configuration will also be saved in csv file.

**Usage**

.. py:function:: stpredict(data, forecast_horizon, history_length = 1, column_identifier = None, feature_sets = {'covariate': 'mRMR'}, models = ['knn'], model_type = 'regression', test_type = 'whole-as-one', mixed_models = [], performance_benchmark = 'MAPE', performance_measures = ['MAPE'], performance_mode = 'normal', splitting_type = 'training-validation', instance_testing_size = 0.2, instance_validation_size = 0.3, instance_random_partitioning = False, fold_total_number = 5, imputation = True, target_mode = 'normal', feature_scaler = None, target_scaler = None, forced_covariates = [], futuristic_covariates = None, scenario = 'current', future_data_table = None, temporal_scale_level = 1, spatial_scale_level = 1, spatial_scale_table = None, aggregation_mode = 'mean', augmentation = False, validation_performance_report = True, testing_performance_report = True, save_predictions = True, save_ranked_features = True, plot_predictions = False, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: stpredict_in.csv

.. Note:: In the current version, 'AIC' and 'BIC' can only be calculated for the 'glm' model and 'classification' model_type.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: stpredict_out.csv

**Example** 

.. code-block:: python

   from stpredict import stpredict
   
   df1 = pd.read_csv('USA COVID-19 temporal data.csv')
   df2 = pd.read_csv('USA COVID-19 spatial data.csv')

   stpredict(data = {'temporal_data':df1,'spatial_data':df2},
                      forecast_horizon = 2, history_length = 2,
                      models = ['knn'], model_type = 'regression')

