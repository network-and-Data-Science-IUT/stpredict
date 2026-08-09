import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import geopandas
import os
from os.path import exists

def get_map(predictions, geometries, spatial_ids = None, temporal_ids = None, model_name = 'knn'):

  # check spatial_ids input
  if spatial_ids is None:
    spatial_ids = np.unique(list(predictions['spatial id']))
  elif not isinstance(spatial_ids, (list,np.ndarray)):
    raise ValueError("spatial_ids input must be a list of spatial ids.")
  number_of_spatial_ids = len(spatial_ids)

  # check geometries input
  if isinstance(geometries, (list,np.ndarray)):
    # if geometries is the list of 2 elements 1.the list of latitudes and 2.the list of longitudes
    if len(geometries)==2 and isinstance(geometries[0], (list,np.ndarray)) and isinstance(geometries[1], (list,np.ndarray)):
      lats = geometries[0]
      longs = geometries[1]
      if not (len(lats) == len(longs) == number_of_spatial_ids):
        raise ValueError("The number of geometries does not match the number of spatial ids.")
      geometries = [shapely.Point(lats[coord_number], longs[coord_number]) for coord_number in range(0,number_of_spatial_ids)]

    # if geometries is a list of tuples (latitude,longitude)
    elif np.all(list(map(lambda item: isinstance(item, tuple) and len(item) == 2, geometries))) :
      if not len(geometries) == number_of_spatial_ids:
        raise ValueError("The number of geometries does not match the number of spatial ids.")
      geometries = [shapely.Point(geometries[coord_number][0], geometries[coord_number][1]) for coord_number in range(0,number_of_spatial_ids)]

  # if geometries is the address of geopandas dataframe
  elif isinstance(geometries, str):
    try:
      geometries = geopandas.read_file(geometries)
    except Exception:
      raise ValueError("There is a problem with reading the shape file.")
    geometries = list(geometries['geometry'])

  elif geometries is None:
    raise ValueError("geometries input must be passed.")
  else:
    raise ValueError("geometries input does not match any of the expected values.")

  # check predictions input
  if isinstance(predictions, str):
    try:
      predictions = pd.read_csv(predictions)
    except Exception:
      raise ValueError("There is a problem with reading the predictions file.")
  elif not isinstance(predictions, pd.DataFrame):
    raise ValueError("predictions input must be a DataFrame or file address of a CSV file.")
  prediction_columns = predictions.columns.values
  if 'spatial id' not in prediction_columns or 'temporal id' not in prediction_columns:
    raise ValueError("predictions input must include the columns: 'spatial id', 'temporal id'.")
  elif ('prediction' not in prediction_columns) and not ('class 0' in prediction_columns and 'class 1' in prediction_columns):
    raise ValueError("predictions input must include the column 'prediction', or columns 'class 0' and 'class 1'.")

  # check temporal_ids input
  if isinstance(temporal_ids, (list,np.ndarray)):
    if len(set(temporal_ids) - set(predictions['temporal id']))>0:
      raise ValueError("Some of the items in the temporal_ids input are not in the predictions file.")
    predictions = predictions.loc[(predictions['temporal id'].isin(temporal_ids))]
  elif temporal_ids is None:
    temporal_ids = np.unique(list(predictions['temporal id']))
  else:
    raise ValueError("temporal_ids input must be a list of temporal ids.")

  # check temporal_ids input
  if model_name not in list(predictions['model name']):
    raise ValueError("model_name input is not valid.")
  predictions = predictions.loc[predictions['model name'] == model_name]

  geometries = pd.DataFrame({'spatial id':spatial_ids,'geometry':geometries})

  # Get final predicted class from the predicted probabilities
  classes = None
  if 'class 0' in predictions.columns.values:
    classes = [item for item in predictions.columns.values if item.startswith('class ')]
    predictions.insert(loc = predictions.shape[1], column='max_class', value=list(predictions[classes].max(axis = 1)))
    predictions.insert(loc = predictions.shape[1], column='predicted_class', value=np.nan)
    for i, class_name in enumerate(sorted(classes)):
      temp_df = predictions.loc[(predictions[class_name]==predictions['max_class'])]
      ind = temp_df[temp_df['predicted_class'].isna()].index
      predictions.loc[ind,'predicted_class'] = class_name
  
  if not exists('./plots'):
            os.makedirs('./plots')

  # plot the map for each temporal id in sepparate file
  for temporal_id in temporal_ids:
    temp = predictions[predictions['temporal id'] == temporal_id]
    geo_data = pd.merge(geometries.copy(), temp, on = 'spatial id', how='left')
    geo_data = geopandas.GeoDataFrame(data = geo_data, geometry = 'geometry', crs='EPSG:4326')

    fig, ax = plt.subplots()
    # regression case
    if 'prediction' in geo_data.columns.values: 
      geo_data.plot(column="prediction", ax=ax, marker='o', legend=True, markersize=5, missing_kwds={'color': 'lightgrey'})
    # classification case
    else: 
      geo_data.plot(column="predicted_class", ax=ax, marker='o', categorical = True, legend=True, markersize=5, missing_kwds={'color': 'lightgrey'})
    
    fig.savefig('./plots/'+str(temporal_id).replace('/','.')+" map.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=1)
    plt.close()