.. _target user_defined_model:

User defined model
==================

User-defined model format
-------------------------

As it's described in the definition of functions, the user-defined models can be accepted as an input to be trained and evaluated in a package functions. The format of these functions must be in accordance with the following description:

**Parameters**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: user_defined_model_format_in.csv

**Returns**

.. csv-table::
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: user_defined_model_format_out.csv



The custom model function must be defined in this format:

.. code-block:: python

   def custom_model_name(X_training, X_validation, Y_training):
        # import required packages
	# define your model
	# train the model on X_training and Y_training
	# predict values on X_training
	# predict values on X_validation
	return(train_predictions, validation_predictions, trained_model)
