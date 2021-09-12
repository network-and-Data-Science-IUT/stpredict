target_modification
===================

**Description**

Modify the target variable based on specified target mode

**Usage**

.. py:function:: preprocess.target_modification(data, target_mode, column_identifier = None, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: target_modification_in.csv


**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: target_modification_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import target_modification

   df = pd.read_csv('USA COVID-19 temporal data.csv')

   modified_df = target_modification(data = df, target_mode = 'moving average')

