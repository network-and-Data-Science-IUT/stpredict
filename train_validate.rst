train_validate
==============

**Description**

training and validation process on preprocessed historical data to find the best model and the corresponding best configurations (best feature set and history length) for it.

**Usage**

.. py:function:: predict.train_validate(data, feature_sets, forced_covariates = [], instance_validation_size = 0.3, instance_testing_size = 0, fold_total_number = 5, instance_random_partitioning = False, forecast_horizon = 1, models = ['knn'], mixed_models = None,  model_type = 'regression', splitting_type = 'training-validation', performance_measures = None, performance_benchmark = None, performance_mode = 'normal', feature_scaler = None, target_scaler = None, labels = None, performance_report = True, save_predictions = True, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: train_validate_in.csv


**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: train_validate_out.csv

**Example** 

.. code-block:: python

   from stpredict.predict import train_validate

   best_model, best_model_parameters, best_history_length, 
   best_feature_or_covariate_set, best_model_base_models, best_trained_model = train_validate(
                       data = ['./historical_data h=1.csv', './historical_data h=2.csv', 
                               './historical_data h=3.csv'],
                       feature_sets =  {'covariate': 'correlation'}, forecast_horizon = 4)



