predict
=======

**Description**

Finding the best history length and feature set based on models performance on the validation set, reporting the performance of the models, for each history length and feature set on the training and validation set, and report the performance of the best model with selected history length and feature set for the test set, and predicting the target variable values for the temporal units in the future.

**Usage**

.. py:function:: predict.predict(data, forecast_horizon,  feature_sets = {'covariate':'mRMR'}, forced_covariates = [], models = ['knn'],  mixed_models = ['knn'], model_type = 'regression', test_type = 'whole-as-one', splitting_type = 'training-validation',  instance_testing_size = 0.2, instance_validation_size = 0.3, instance_random_partitioning = False, fold_total_number = 5, feature_scaler = None, target_scaler = None, performance_benchmark = 'MAPE',  performance_measure = ['MAPE'], performance_mode = 'normal', scenario = ‘current’, validation_performance_report = True, testing_performance_report = True, save_predictions = True, save_ranked_features = True, plot_predictions = False, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: predict.predict_in.csv

.. Note:: In the current version, 'AIC' and 'BIC' can only be calculated for the 'glm' model and 'classification' model_type.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: predict.predict_out.csv

**Example** 

.. code-block:: python

   from stpredict.predict import predict
   
   predict(data = ['./historical_data h=1.csv', './historical_data h=2.csv',
                   './historical_data h=3.csv'], forecast_horizon = 4)

