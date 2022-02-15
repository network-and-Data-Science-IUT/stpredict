import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import ParameterGrid
    from sklearn.neural_network import MLPRegressor
    from sklearn import linear_model
    from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LogisticRegression
    from sklearn import svm
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
    import time
    from sys import argv
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import activations
    from tensorflow.keras.callbacks import EarlyStopping
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.wrappers.scikit_learn import KerasClassifier
    import os
    from numpy.random import seed
    from keras import backend as K
    import random
    
seed(1)
tf.random.set_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# reset seed for reprodutability of neural network
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
        else:raise ValueError(error_msg)
        # remove duplicate information on network structure
        user_params = {key:user_params[key] for key in user_params.keys() if key not in ['hidden_layers_neurons', 'hidden_layers_activations', 'hidden_layers_number']}

    else:
        # extract hidden_layers_structure from hidden_layers_neurons, hidden_layers_activations and hidden_layers_number
        if 'hidden_layers_neurons' not in user_params:
            user_params['hidden_layers_neurons'] = None
        elif type(user_params['hidden_layers_neurons']) != int:
            raise TypeError('The value of hidden_layers_neurons in NN model parameters must be of type int.')

        if 'hidden_layers_activations' not in user_params:
            user_params['hidden_layers_activations'] = None
        elif (user_params['hidden_layers_activations'] is not None) and (type(user_params['hidden_layers_activations'])!=str):
            raise TypeError('The value of hidden_layers_activations in NN model parameters must be of type string or None.')
            
        if 'hidden_layers_number' not in user_params:
            user_params['hidden_layers_number'] = 1
        elif type(user_params['hidden_layers_number']) != int:
            raise TypeError('The value of hidden_layers_number in NN model parameters must be of type int.')

        user_params['hidden_layers_structure'] = []
        for layer in range(1,user_params['hidden_layers_number']+1):
            user_params['hidden_layers_structure'].append(tuple((user_params['hidden_layers_neurons'],user_params['hidden_layers_activations'])))

        # remove duplicate information on network structure
        user_params = {key:user_params[key] for key in user_params.keys() if key 
                       not in ['hidden_layers_neurons', 'hidden_layers_activations', 'hidden_layers_number']}
    
    return user_params
    

####################################################### GBM: Gradient Boosting Regressor
def GBM_REGRESSOR(X_train, X_test, y_train, user_params, verbose):

    parameters = {'loss':'ls', 'learning_rate':0.1, 'n_estimators':100, 'subsample':1.0, 'criterion':'friedman_mse', 
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
    
    parameters = {'alpha':1.0, 'l1_ratio':0.5, 'fit_intercept':True, 'normalize':False, 'precompute':False,
                  'max_iter':1000, 'copy_X':True, 'tol':0.0001, 'warm_start':False, 'positive':False, 'random_state':1,
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
            else: raise ValueError('The number of neighbors in the knn model parameters must be of type int.')
                
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
    y_prediction_train = GradientBoostingclassifierObject.predict_proba(X_train)

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
    y_prediction_train = GLM_Model.predict_proba(X_train)
    

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
            else: raise ValueError('The number of neighbors in the knn model parameters must be of type int.')
                
    if valid_k_flag == 0:
        KNeighborsClassifierObject = KNeighborsClassifier()
        # Grid search over different Ks to choose the best one
        neighbors=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200])
        neighbors=neighbors[neighbors<len(X_train)*(4/5)] #4/5 of samples is used as train when cv=5
        grid_parameters = {'n_neighbors': neighbors}
        GridSearchOnKs = GridSearchCV(KNeighborsClassifierObject, grid_parameters, cv=5)
        GridSearchOnKs.fit(X_train, y_train)
        best_K = GridSearchOnKs.best_params_
        
        
        if verbose == 1:
            print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
            print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]")
            print('best k:', best_K['n_neighbors'])
            
        K = best_K['n_neighbors']
        
    KNN_Model = KNeighborsClassifier(n_neighbors=K, **parameters)
    KNN_Model.fit(X_train, y_train.ravel())
    y_prediction = KNN_Model.predict_proba(X_test)
    y_prediction_train = KNN_Model.predict_proba(X_train)

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
    dummy_y_train = np_utils.to_categorical(encoded_y_train)
    
    
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

