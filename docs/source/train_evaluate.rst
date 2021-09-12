train_evaluate
==============

**Description**

training the models on the training set and predict the target variable values for the training and validation set.

**Usage**

.. py:function:: preprocess.train_evaluate(training_data, validation_data, model, model_type, model_parameters = None, labels=None, base_models = None, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 5, 10
   :file: train_evaluate_in.csv

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 5, 10
   :file: train_evaluate_out.csv

**Example** 

.. code-block:: Python

   import pandas as pd
   from stpredict.predict import train_evaluate
   
   df = pd.read_csv('./historical_data h=1.csv')
   training_df = df.iloc[:-200]
   validation_df = df.iloc[-200:]
   
   train_predictions, validation_predictions, trained_model = train_evaluate(
                      training_data = training_df, validation_data = validation_df, model = 'nn',
                      model_parameters = {'hidden_layers_structure':[(2,None),(4,'relu'),(8,'relu')]})

