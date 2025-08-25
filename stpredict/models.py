import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    from sklearn.experimental import enable_hist_gradient_boosting # noqa
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import ElasticNet, LogisticRegression
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers # noqa
    from tensorflow.keras import activations # noqa
    from tensorflow.keras.callbacks import EarlyStopping
    from keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    from .sgcrf import SparseGaussianCRF, SparseGaussianCRFclassifier
    from collections import Counter
    import os
    from numpy.random import seed
    from keras import backend as K
    import random

seed(1)
tf.random.set_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# reset seed for reproducibility of neural network
def reset_seeds():
    np.random.seed(1)
    random.seed(1)
    if tf.__version__[0] == '2':
        tf.random.set_seed(1)
    else:
        tf.set_random_seed(1)

# producing list of parameter values combinations from parameter grid specified by user
def get_nn_structure(user_params):

    if user_params is None:
        return user_params

    if 'hidden_layers_structure' in user_params:
        error_msg = 'The value of hidden_layers_structure in NN model parameters must be a list of tuples including number of neurons and activation function of each layer.'
        if not isinstance(user_params['hidden_layers_structure'],list):
            raise ValueError(error_msg)
        elif all([isinstance(item, tuple) for item in user_params['hidden_layers_structure']]):
            if not all([len(item) == 2 for item in user_params['hidden_layers_structure']]):
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
        # remove duplicate information on network structure
        user_params = {key:user_params[key] for key in user_params.keys() if key not in ['hidden_layers_neurons', 'hidden_layers_activations', 'hidden_layers_number']}

    else:
        # extract hidden_layers_structure from hidden_layers_neurons, hidden_layers_activations and hidden_layers_number
        if 'hidden_layers_neurons' not in user_params:
            user_params['hidden_layers_neurons'] = None
        elif not isinstance(user_params['hidden_layers_neurons'], int):
            raise TypeError('The value of hidden_layers_neurons in NN model parameters must be of type int.')

        if 'hidden_layers_activations' not in user_params:
            user_params['hidden_layers_activations'] = None
        elif (user_params['hidden_layers_activations'] is not None) and (not isinstance(user_params['hidden_layers_activations'], str)):
            raise TypeError('The value of hidden_layers_activations in NN model parameters must be of type string or None.')

        if 'hidden_layers_number' not in user_params:
            user_params['hidden_layers_number'] = 1
        elif not isinstance(user_params['hidden_layers_number'], int):
            raise TypeError('The value of hidden_layers_number in NN model parameters must be of type int.')

        user_params['hidden_layers_structure'] = []
        for layer in range(1,user_params['hidden_layers_number']+1):
            user_params['hidden_layers_structure'].append(tuple((user_params['hidden_layers_neurons'],user_params['hidden_layers_activations'])))

        # remove duplicate information on network structure
        user_params = {key:user_params[key] for key in user_params.keys() if key
                       not in ['hidden_layers_neurons', 'hidden_layers_activations', 'hidden_layers_number']}

    return user_params
    
def get_sgcrf_data(X_train, X_test, y_train, train_ids, test_ids, neighbouring_list, number_of_layers, classification):
    
    spatial_ids = np.array(train_ids['spatial id'].unique())
    features = list(X_train.columns.values)

    # Adding ids to the features and output values
    X_train = pd.concat([train_ids.reset_index(drop = True), X_train.reset_index(drop = True)], axis=1)
    X_train = X_train.sort_values(by = ['temporal id','spatial id'])
    
    X_test = pd.concat([test_ids.reset_index(drop = True), X_test.reset_index(drop = True)], axis=1)
    X_test = X_test.sort_values(by = ['temporal id','spatial id'])

    y_train_df = pd.DataFrame({'Target':list(y_train)})
    y_train_df = pd.concat([train_ids.reset_index(drop = True), y_train_df.reset_index(drop = True)], axis=1)
    y_train_df = y_train_df.sort_values(by = ['temporal id','spatial id'])
    
    #y_train_df.to_csv('y_train_df.csv')
    #X_test.to_csv('X_test.csv')
    #X_train.to_csv('X_train.csv')
    np.save('neighbouring_list.npy',neighbouring_list)
    layer_matrix_dict = {}
    layer_matrix_dict[1] = neighbouring_list
    max_num_neighbours_dict = {}
    max_num_neighbours_dict[1] = 0

    # Finding neighbouring layers' adjacency matrix
    for layer in range(1,number_of_layers+1):
            if layer > 1:
                layer_matrix_dict[layer] = {}
                max_num_neighbours_dict[layer] = 0
            for spatial_id in spatial_ids:
                if layer > 1:
                    temp = []
                    # Add the neighbours of previous layer (l-1) as l level neighbours
                    for neighbour in layer_matrix_dict[layer-1][spatial_id]:
                        temp = temp + layer_matrix_dict[1][neighbour]
                    # Remove the neighbours that are already in the previous layers
                    for previous_layer in range(1,layer):
                        temp = list(set(temp) - set(layer_matrix_dict[previous_layer][spatial_id]))
                    # Remove the self neighbourhood
                    if spatial_id in temp:
                        temp.remove(spatial_id)
                    layer_matrix_dict[layer][spatial_id] = temp

                    if len(temp) > max_num_neighbours_dict[layer]:
                        max_num_neighbours_dict[layer] = len(temp)
                else:
                    neighbours = layer_matrix_dict[layer][spatial_id]
                    if len(neighbours) > max_num_neighbours_dict[layer]:
                        max_num_neighbours_dict[layer] = len(neighbours)

    # initialize model data
    X_train_final = pd.DataFrame()
    X_test_final = pd.DataFrame()
    y_train_final = pd.DataFrame()


    # get neighbouring data for each spatial_id in each layer
    for spatial_id in spatial_ids:

        X_train_temp = X_train[X_train['spatial id']==spatial_id]
        X_train_temp = X_train_temp.sort_values(by = ['temporal id'])
        X_train_temp = X_train_temp.drop(['spatial id','temporal id'], axis = 1)

        X_test_temp = X_test[X_test['spatial id']==spatial_id]
        X_test_temp = X_test_temp.sort_values(by = ['temporal id'])
        X_test_temp = X_test_temp.drop(['spatial id','temporal id'], axis = 1)

        y_train_temp = y_train_df[y_train_df['spatial id']==spatial_id]
        y_train_temp = y_train_temp.sort_values(by = ['temporal id'])
        y_train_temp = y_train_temp.drop(['spatial id','temporal id'], axis = 1)


        for layer in range(1,number_of_layers+1):

            max_num_neighbours = max_num_neighbours_dict[layer]

            neighbours = layer_matrix_dict[layer][spatial_id]
            dummy_neighbours_num = max_num_neighbours - len(neighbours)

            X_train_neighbours_data = X_train[X_train['spatial id'].isin(neighbours)]
            X_test_neighbours_data = X_test[X_test['spatial id'].isin(neighbours)]
            y_train_neighbours_data = y_train_df[y_train_df['spatial id'].isin(neighbours)]

            # Create dummy zero-filled neighbouring data to make the row of this spatial unit match other rows
            if len(neighbours) == 0:

                X_train_dummy_neighbour_data = pd.DataFrame(columns=X_train.columns)
                X_train_dummy_neighbour_data['temporal id'] = list(X_train['temporal id'].unique())
                X_train_dummy_neighbour_data = X_train_dummy_neighbour_data.fillna(0)

                X_test_dummy_neighbour_data = pd.DataFrame(columns=X_test.columns)
                X_test_dummy_neighbour_data['temporal id'] = list(X_test['temporal id'].unique())
                X_test_dummy_neighbour_data = X_test_dummy_neighbour_data.fillna(0)

                y_train_dummy_neighbour_data = pd.DataFrame(columns=y_train_df.columns)
                y_train_dummy_neighbour_data['temporal id'] = list(y_train_df['temporal id'].unique())
                y_train_dummy_neighbour_data = y_train_dummy_neighbour_data.fillna(0)

                # y should be lable for classification so we randomly sample labels
                if classification == 1:
                    y_train_dummy_neighbour_data['Target'] = list(random.sample(list(y_train_df['Target']),len(y_train_dummy_neighbour_data)))

            # Create dummy neighbouring data to make the row of this spatial unit match other rows.
            # It fills with the average values of the other neighbours.
            elif len(neighbours) < max_num_neighbours:

                X_train_dummy_neighbour_data = X_train_neighbours_data.groupby(['temporal id']).mean()
                X_train_dummy_neighbour_data = X_train_dummy_neighbour_data.reset_index().reindex(columns=X_train_neighbours_data.columns)

                X_test_dummy_neighbour_data = X_test_neighbours_data.groupby(['temporal id']).mean()
                X_test_dummy_neighbour_data = X_test_dummy_neighbour_data.reset_index().reindex(columns=X_test_neighbours_data.columns)

                y_train_dummy_neighbour_data = y_train_neighbours_data.groupby(['temporal id']).mean()
                y_train_dummy_neighbour_data = y_train_dummy_neighbour_data.reset_index().reindex(columns=y_train_neighbours_data.columns)

                if classification == 1:
                    y_train_dummy_neighbour_data.loc[y_train_dummy_neighbour_data['Target']>=0.5,('Target')]=1
                    y_train_dummy_neighbour_data.loc[y_train_dummy_neighbour_data['Target']<0.5,('Target')]=0

            # Repeatedly add dummy neighbor as needed
            for i in range(dummy_neighbours_num):
                X_train_dummy_neighbour_data['spatial id'] = 'dummy spatial id level '+str(layer)+' '+str(i)
                X_test_dummy_neighbour_data['spatial id'] = 'dummy spatial id level '+str(layer)+' '+str(i)
                y_train_dummy_neighbour_data['spatial id'] = 'dummy spatial id level '+str(layer)+' '+str(i)

                if len(neighbours) == 0 and i == 0:
                    X_train_neighbours_data = X_train_dummy_neighbour_data.copy()
                    X_test_neighbours_data = X_test_dummy_neighbour_data.copy()
                    y_train_neighbours_data = y_train_dummy_neighbour_data.copy()
                else:
                    X_train_neighbours_data = pd.concat([X_train_neighbours_data, X_train_dummy_neighbour_data])
                    X_test_neighbours_data = pd.concat([X_test_neighbours_data, X_test_dummy_neighbour_data])
                    y_train_neighbours_data = pd.concat([y_train_neighbours_data, y_train_dummy_neighbour_data])

            # Pivot neighbouring data and add it to the rows of this spatial unit. Finaly we get one row of 
            # feature values for the spatial unit and all its neighbours for each temporal unit.
            X_train_neighbours_data = X_train_neighbours_data.sort_values(by = ['temporal id','spatial id'])
            X_train_neighbours_data = X_train_neighbours_data.pivot(index='temporal id', columns='spatial id', values=features)
            X_train_neighbours_data.columns = X_train_neighbours_data.columns.map('{0[0]}|{0[1]}'.format)
            X_train_neighbours_data = X_train_neighbours_data.sort_index()

            X_test_neighbours_data = X_test_neighbours_data.sort_values(by = ['temporal id','spatial id'])
            X_test_neighbours_data = X_test_neighbours_data.pivot(index='temporal id', columns='spatial id', values=features)
            X_test_neighbours_data.columns = X_test_neighbours_data.columns.map('{0[0]}|{0[1]}'.format)
            X_test_neighbours_data = X_test_neighbours_data.sort_index()

            y_train_neighbours_data = y_train_neighbours_data.sort_values(by = ['temporal id','spatial id'])
            y_train_neighbours_data = y_train_neighbours_data.pivot(index='temporal id', columns='spatial id', values='Target')
            y_train_neighbours_data = y_train_neighbours_data.sort_index()


            X_train_temp = pd.concat([X_train_temp.reset_index(drop = True),X_train_neighbours_data.reset_index(drop = True)],axis=1)
            X_test_temp = pd.concat([X_test_temp.reset_index(drop = True),X_test_neighbours_data.reset_index(drop = True)],axis=1)
            y_train_temp = pd.concat([y_train_temp.reset_index(drop = True),y_train_neighbours_data.reset_index(drop = True)],axis=1)

        # adding rows of this spatial unit to the whole training and testing data
        X_train_temp.columns = list(range(len(X_train_temp.columns)))
        X_test_temp.columns = list(range(len(X_test_temp.columns)))
        y_train_temp.columns = list(range(len(y_train_temp.columns)))

        X_train_final = pd.concat([X_train_final,X_train_temp],axis=0)
        X_test_final = pd.concat([X_test_final,X_test_temp],axis=0)
        y_train_final = pd.concat([y_train_final,y_train_temp],axis=0)    
        #X_train_final.to_csv('X_train_final.csv')
        #X_test_final.to_csv('X_test_final.csv')
        #y_train_final.to_csv('y_train_final.csv')
        
    return(np.array(X_train_final), np.array(X_test_final), np.array(y_train_final))


def resort_classes(probabilities, classes):
  
  sorted_probabilities = np.zeros_like(probabilities)
  sorted_classes = sorted(classes)
  
  for i, class_name in  enumerate(classes):
    sorted_probabilities[:,sorted_classes.index(class_name)] = probabilities[:,i]
  return(sorted_probabilities)

####################################################### GBM: Gradient Boosting Regressor
def GBM_REGRESSOR(X_train, X_test, y_train, user_params, verbose):

    parameters = {'loss':'squared_error', 'learning_rate':0.1, 'n_estimators':100, 'subsample':1.0, 'criterion':'friedman_mse',
                  'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_depth':3,
                  'min_impurity_decrease':0.0, 'init':None, 'random_state':None,
                  'max_features':None, 'alpha':0.9, 'verbose':0, 'max_leaf_nodes':None, 'warm_start':False,
                  'validation_fraction':0.1, 'n_iter_no_change':None, 'tol':0.0001, 'ccp_alpha':0.0}


    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    GradientBoostingRegressorObject = GradientBoostingRegressor(**parameters)

    GradientBoostingRegressorObject.fit(X_train, y_train)
    y_prediction = GradientBoostingRegressorObject.predict(X_test)
    y_prediction_train = GradientBoostingRegressorObject.predict(X_train)


    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GradientBoostingRegressorObject

###################################################### GLM: Generalized Linear Model Regressor
def GLM_REGRESSOR(X_train, X_test, y_train, user_params, verbose):

    parameters = {'alpha':1.0, 'l1_ratio':0.5, 'fit_intercept':True, 'precompute':False,
                  'max_iter':1000, 'copy_X':True, 'tol':0.0001, 'warm_start':False, 'positive':False, 'random_state':None,
                  'selection':'cyclic'}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    GLM_Model = ElasticNet(**parameters)
    GLM_Model.fit(X_train, y_train)
    y_prediction = GLM_Model.predict(X_test)
    y_prediction_train = GLM_Model.predict(X_train)

    if verbose == 1:
        print('GLM coef: ', GLM_Model.coef_)

    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GLM_Model


######################################################### KNN: K-Nearest Neighbors Regressor
def KNN_REGRESSOR(X_train, X_test, y_train, user_params, verbose):

    parameters = {'weights':'uniform', 'algorithm':'auto', 'leaf_size':30, 'p':2, 'metric':'minkowski',
                  'metric_params':None, 'n_jobs':None}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    # if user does not specify the K parameter or specified value is too large, the best k will be obtained using a grid search
    valid_k_flag = 0
    if user_params is not None:
        if ('n_neighbors' in user_params.keys()):
            if isinstance(user_params['n_neighbors'],int):
                if (user_params['n_neighbors']<len(X_train)):
                    K = user_params['n_neighbors']
                    valid_k_flag = 1
            else:
                raise ValueError('The number of neighbors in the knn model parameters must be of type int.')

    if valid_k_flag == 0:
        KNeighborsRegressorObject = KNeighborsRegressor()
        # Grid search over different Ks to choose the best one
        neighbors=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200])
        neighbors=neighbors[neighbors<len(X_train)*(4/5)] #4/5 of samples is used as train when cv=5
        grid_parameters = {'n_neighbors': neighbors}
        GridSearchOnKs = GridSearchCV(KNeighborsRegressorObject, grid_parameters, cv=5)
        GridSearchOnKs.fit(X_train, y_train)
        best_K = GridSearchOnKs.best_params_

        if verbose == 1:
            print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
            print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]")
            print('best k:', best_K['n_neighbors'])

        K = best_K['n_neighbors']

    KNN_Model = KNeighborsRegressor(n_neighbors=K, **parameters)
    KNN_Model.fit(X_train, y_train)
    y_prediction = KNN_Model.predict(X_test)
    y_prediction_train = KNN_Model.predict(X_train)

    return y_prediction, y_prediction_train, KNN_Model


####################################################### NN: Neural Network Regressor
def NN_REGRESSOR(X_train, X_test, y_train, user_params, verbose):
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    reset_seeds()

    user_params = get_nn_structure(user_params)

    # default parameters
    parameters = {'hidden_layers_structure':[((X_train.shape[1]) // 2 + 1, None)], 'output_activation':'exponential',
                  'loss':'mean_squared_error',
                  'optimizer':'RMSprop', 'metrics':['mean_squared_error'],
                  'early_stopping_monitor':'val_loss', 'early_stopping_patience':30, 'batch_size':128,
                  'validation_split':0.2,'epochs':100}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    NeuralNetworkObject = keras.models.Sequential()
    NeuralNetworkObject.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
    for neurons, activation in parameters['hidden_layers_structure']:
        neurons = (X_train.shape[1]) // 2 + 1 if neurons is None else neurons
        NeuralNetworkObject.add(tf.keras.layers.Dense(neurons, activation=activation))
    NeuralNetworkObject.add(tf.keras.layers.Dense(1, activation=parameters['output_activation']))


    # Compile the model
    NeuralNetworkObject.compile(
        loss=parameters['loss'],
        optimizer=parameters['optimizer'],
        metrics=parameters['metrics'])

    early_stop = EarlyStopping(monitor=parameters['early_stopping_monitor'], patience=parameters['early_stopping_patience'])

    NeuralNetworkObject.fit(X_train, y_train.ravel(),
                   callbacks=[early_stop],
                   batch_size=parameters['batch_size'],
                   validation_split=parameters['validation_split'],
                   epochs=parameters['epochs'], verbose=0)


    y_prediction = NeuralNetworkObject.predict(X_test)
    y_prediction_train = NeuralNetworkObject.predict(X_train)

    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), NeuralNetworkObject

######################################################### SGCRF: Sparse Gaussian Conditional Random Field Regressor
def SGCRF_REGRESSOR(X_train, X_test, y_train, user_params, data_ids, verbose):

    reset_seeds()

    parameters = {'learning_rate':1.0, 'n_iter':1000, 'lamL':0.01, 'lamT':0.01,
                  'neighbouring_matrix':[], 'number_of_layers':1}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    # Saving the current sorting of the rows
    train_ids = data_ids['training']
    train_ids['sort'] = list(range(len(train_ids)))
    test_ids = data_ids['validation']
    test_ids['sort'] = list(range(len(test_ids)))

    
    # Create input and output matrices for train and test sets
    X_train, X_test, Y_train = get_sgcrf_data(X_train, X_test, y_train, train_ids, test_ids, parameters['neighbouring_matrix'], 
                                              parameters['number_of_layers'], 0)
    
    del parameters['neighbouring_matrix'], parameters['number_of_layers']

    # Fit the model and make prediction
    SGCRF_Model = SparseGaussianCRF(**parameters)
    SGCRF_Model.fit(X_train, Y_train)
    test_prediction_matrice = SGCRF_Model.predict(X_test)
    train_prediction_matrice = SGCRF_Model.predict(X_train)

    # Manipulating prediction arrays to get their initial order
    test_prediction_matrice = test_prediction_matrice[:,1]
    train_prediction_matrice = train_prediction_matrice[:,1]
    train_ids = train_ids.sort_values(by = ['spatial id','temporal id'])
    test_ids = test_ids.sort_values(by = ['spatial id','temporal id'])
    train_ids['prediction'] = list(train_prediction_matrice)
    test_ids['prediction'] = list(test_prediction_matrice)
    train_ids = train_ids.sort_values(by = 'sort')
    test_ids = test_ids.sort_values(by = 'sort')
    y_prediction = list(test_ids['prediction'])
    y_prediction_train = list(train_ids['prediction'])

    return y_prediction, y_prediction_train, SGCRF_Model


####################################################### GBM: Gradient Boosting Classifier

def GBM_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):

    parameters = {'loss':'deviance', 'learning_rate':0.1, 'n_estimators':100, 'subsample':1.0, 'criterion':'friedman_mse',
                  'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_depth':3, 'min_impurity_decrease':0.0,
                  'init':None, 'random_state':1, 'max_features':None, 'verbose':0, 'max_leaf_nodes':None,
                  'warm_start':False, 'validation_fraction':0.1, 'n_iter_no_change':None, 'tol':0.0001, 'ccp_alpha':0.0}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    GradientBoostingclassifierObject = GradientBoostingClassifier(**parameters)

    GradientBoostingclassifierObject.fit(X_train, y_train.ravel())
    y_prediction = GradientBoostingclassifierObject.predict_proba(X_test)
    y_prediction = resort_classes(y_prediction, GradientBoostingclassifierObject.classes_)
    y_prediction_train = GradientBoostingclassifierObject.predict_proba(X_train)
    y_prediction_train = resort_classes(y_prediction_train, GradientBoostingclassifierObject.classes_)

    return y_prediction, y_prediction_train, GradientBoostingclassifierObject


##################################################### GLM: Generalized Linear Model Classifier

def GLM_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):

    parameters = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1,
                  'class_weight':None, 'random_state':1, 'solver':'lbfgs', 'max_iter':100, 'multi_class':'auto',
                  'verbose':0, 'warm_start':False, 'n_jobs':None, 'l1_ratio':None}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    GLM_Model = LogisticRegression(**parameters)
    GLM_Model.fit(X_train, y_train.ravel())
    y_prediction = GLM_Model.predict_proba(X_test)
    y_prediction = resort_classes(y_prediction, GLM_Model.classes_)
    y_prediction_train = GLM_Model.predict_proba(X_train)
    y_prediction_train = resort_classes(y_prediction_train, GLM_Model.classes_)

    return y_prediction, y_prediction_train, GLM_Model

######################################################### KNN: K-Nearest Neighbors Classifier
def KNN_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):

    parameters = {'weights':'uniform', 'algorithm':'auto', 'leaf_size':30, 'p':2,
                  'metric':'minkowski', 'metric_params':None, 'n_jobs':None}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    # if user does not specify the K parameter or specified value is too large, the best k will be obtained using a grid search
    valid_k_flag = 0
    if user_params is not None:
        if ('n_neighbors' in user_params.keys()):
            if isinstance(user_params['n_neighbors'],int):
                if (user_params['n_neighbors']<len(X_train)):
                    K = user_params['n_neighbors']
                    valid_k_flag = 1
            else:
                raise ValueError('The number of neighbors in the knn model parameters must be of type int.')

    if valid_k_flag == 0:
        KNeighborsClassifierObject = KNeighborsClassifier()
        # Grid search over different Ks to choose the best one
        neighbors=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200])
        neighbors=neighbors[neighbors<len(X_train)*(4/5)] #4/5 of samples is used as train when cv=5
        grid_parameters = {'n_neighbors': neighbors}
        count = Counter(y_train)
        min_number_of_class_members = min(list(count.values()))
        cv = 5 if min_number_of_class_members>=5 else min_number_of_class_members
        if cv == 1:
            K = 3 if len(X_train)>3 else len(X_train)
            if verbose == 1:
                print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
                print(f"The number of neighbors will be set to {K}.")
        else:
            GridSearchOnKs = GridSearchCV(KNeighborsClassifierObject, grid_parameters, cv=cv)
            GridSearchOnKs.fit(X_train, y_train)
            best_K = GridSearchOnKs.best_params_
            K = best_K['n_neighbors']

            if verbose == 1:
                print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
                print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]")
                print('best k:', K)
    
            

    KNN_Model = KNeighborsClassifier(n_neighbors=K, **parameters)
    KNN_Model.fit(X_train, y_train.ravel())
    y_prediction = KNN_Model.predict_proba(X_test)
    y_prediction_train = KNN_Model.predict_proba(X_train)
    # classes are already sorted

    return y_prediction, y_prediction_train, KNN_Model

    
####################################################### NN: Neural Network Classifier

def NN_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):

    K.clear_session()
    tf.compat.v1.reset_default_graph()
    reset_seeds()

    user_params = get_nn_structure(user_params)

    # default parameters
    parameters = {'hidden_layers_structure':[((X_train.shape[1]) // 2 + 1, None)],
                  'output_activation':'softmax', 'loss':'categorical_crossentropy',
                  'optimizer':'adam', 'metrics':['accuracy'],
                  'early_stopping_monitor':'val_loss', 'early_stopping_patience':30, 'batch_size':128,
                  'validation_split':0.2,'epochs':100}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    encoder = LabelEncoder().fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    number_of_classes = len(encoder.classes_)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = to_categorical(encoded_y_train)


    output_neurons = number_of_classes

    y_to_fit = dummy_y_train

    if (parameters['output_activation'] == 'sigmoid') and (number_of_classes == 2):
        output_neurons = 1
        y_to_fit = encoded_y_train.ravel()

    NeuralNetworkObject = keras.models.Sequential()
    NeuralNetworkObject.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
    for neurons, activation in parameters['hidden_layers_structure']:
        neurons = (X_train.shape[1]) // 2 + 1 if neurons is None else neurons
        NeuralNetworkObject.add(tf.keras.layers.Dense(neurons, activation=activation))
    NeuralNetworkObject.add(tf.keras.layers.Dense(output_neurons, activation=parameters['output_activation']))


    # Compile the model
    NeuralNetworkObject.compile(
        loss=parameters['loss'],
        optimizer=parameters['optimizer'],
        metrics=parameters['metrics'])

    early_stop = EarlyStopping(monitor=parameters['early_stopping_monitor'], patience=parameters['early_stopping_patience'])


    NeuralNetworkObject.fit(X_train, y_to_fit,
                   callbacks=[early_stop],
                   batch_size=parameters['batch_size'],
                   validation_split=parameters['validation_split'],
                   epochs=parameters['epochs'], verbose=0)

    if (parameters['output_activation'] == 'sigmoid') and (number_of_classes == 2):
        y_prediction = NeuralNetworkObject.predict(X_test)
        y_prediction = np.array([[1-(x[0]),x[0]] for x in y_prediction]).astype("float32")
        y_prediction_train = NeuralNetworkObject.predict(X_train)
        y_prediction_train = np.array([[1-(x[0]),x[0]] for x in y_prediction_train]).astype("float32")

    else:
        y_prediction = NeuralNetworkObject.predict(X_test)
        y_prediction_train = NeuralNetworkObject.predict(X_train)

    return y_prediction, y_prediction_train, NeuralNetworkObject

######################################################### SGCRF: Sparse Gaussian Conditional Random Field Classifier
def SGCRF_CLASSIFIER(X_train, X_test, y_train, user_params, data_ids, verbose):

    reset_seeds()

    parameters = {'learning_rate':1.0, 'n_iter':1000, 'lamL':0.01, 'lamT':0.01,
                  'neighbouring_matrix':[], 'number_of_layers':1}

    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]

    # Saving the current sorting of the rows
    train_ids = data_ids['training']
    train_ids['sort'] = list(range(len(train_ids)))
    test_ids = data_ids['validation']
    test_ids['sort'] = list(range(len(test_ids)))
    
    X_train, X_test, Y_train = get_sgcrf_data(X_train, X_test, y_train, train_ids, test_ids, 
                                              parameters['neighbouring_matrix'][0], 
                                              parameters['number_of_layers'], 1)
    
    del parameters['neighbouring_matrix'], parameters['number_of_layers']

    # Fit the model and make prediction
    SGCRF_Model = SparseGaussianCRFclassifier(**parameters)
    SGCRF_Model.fit(X_train, Y_train)
    test_prediction_matrice = SGCRF_Model.predict(X_test)
    train_prediction_matrice = SGCRF_Model.predict(X_train)

    # Manipulating prediction arrays to get their initial order
    test_prediction_matrice = test_prediction_matrice[:,1]
    train_prediction_matrice = train_prediction_matrice[:,1]
    train_ids = train_ids.sort_values(by = ['spatial id','temporal id'])
    test_ids = test_ids.sort_values(by = ['spatial id','temporal id'])
    train_ids['class 0'] = list(train_prediction_matrice)
    test_ids['class 0'] = list(test_prediction_matrice)
    train_ids.insert(loc=train_ids.shape[1], column='class 1', value = list(map(lambda x: 1-x, train_prediction_matrice)))
    test_ids.insert(loc=test_ids.shape[1], column='class 1', value = list(map(lambda x: 1-x, test_prediction_matrice)))
    train_ids = train_ids.sort_values(by = 'sort')
    test_ids = test_ids.sort_values(by = 'sort')
    y_prediction = np.array(test_ids[['class 0','class 1']])
    y_prediction_train = np.array(train_ids[['class 0','class 1']])

    return y_prediction, y_prediction_train, SGCRF_Model