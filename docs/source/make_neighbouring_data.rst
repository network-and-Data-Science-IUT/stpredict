make_neighbouring_data
======================

**Description**
Extracting new covariates from the data using spatial correlation. To extract a new covariate, for each spatial unit, the values of an existing covariate (spatial or temporal) are averaged in the neighboring spatial units of this unit. It can be repeated for several neighbouring layers, where each layer includes the neighbouring spatial units with a certain distance. The first layer contains adjacent neighbours, the second layer contains neighbours with a distance of one spatial unit, and so on for other layers.


**Usage**

.. py:function:: preprocess.make_neighbouring_data(data, column_identifier = None, number_of_layers = 1, neighbouring_matrix = None, time_dependency_flag = 1, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: make_neighbouring_data_in.csv

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: make_neighbouring_data_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import make_neighbouring_data

   df = pd.read_csv('USA COVID-19 temporal data.csv')


   neighbouring_data = make_neighbouring_data(data = df, number_of_layers = 2,
                                                neighbouring_matrix = [[0,1,...,0],
                                                                       [1,0,...,1],
                                                                       ...,
                                                                       [0,1,...,0]])


