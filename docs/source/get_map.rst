get_map
=======

**Description**

Display the values (or classes) predicted by the predict function on a map and save the map in the sub directory '/plots' in the same directory as the code is running and as in '.pdf' format. 

**Usage**

.. py:function:: predict.get_map(predictions, geometries, spatial_ids = None, temporal_ids = None, model_name = 'knn')

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: get_map_in.csv


**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: stpredict_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.predict import get_map
   
   predictions = pd.read_csv('./predictions/test process/test prediction forecast horizon = 4.csv')
   get_map(predictions = predictions, geometries = 'geometries.shp', 
           spatial_ids = ['Alabama', 'Florida'], temporal_ids = ['2/1/2020'], model_name = 'knn')
