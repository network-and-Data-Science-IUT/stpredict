import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import sys
    import datetime
    import traceback
    from .models import KNN_REGRESSOR, KNN_CLASSIFIER, NN_REGRESSOR, NN_CLASSIFIER, GLM_REGRESSOR, GLM_CLASSIFIER, \
        GBM_REGRESSOR, GBM_CLASSIFIER


def complete_predicted_probabilities(predictions, all_labels, present_labels):
    
    present_probabilities = predictions
    all_probabilities = np.zeros([len(present_probabilities),len(all_labels)])
    ind = 0
    for label_number, label in enumerate(all_labels):
        if label in present_labels:
            all_probabilities[:,label_number] = present_probabilities[:,ind]
            ind += 1

        else:
            all_probabilities[:,label_number] = 0
            
    return all_probabilities

def run_model(X_training, X_validation, Y_training, model, model_type, supported_models_name, model_parameters=None,\
              labels=None, verbose=0):
    
    # Labels presented in input training data
    present_labels = list(np.unique(Y_training))
    present_labels.sort()
    
    # reading user defined models from dump file 'user_defined_models.py', where they have already been saved in train_validate function
    if (type(model) == str)  and (model not in supported_models_name and model not in ['mixed_' + name for name in supported_models_name]):
        try:
            from . import user_defined_models
            model = getattr(user_defined_models, model)
        except Exception:
            raise ValueError(
            "The model must be name of one of the supported models: {'gbm', 'glm', 'knn', 'nn'} Or user defined function.")

    if type(model) == str:
        try:
            if model == 'gbm' or model == 'mixed_gbm':

                if model_type == 'classification':
                    validation_predictions, train_predictions, trained_model = GBM_CLASSIFIER(X_training, X_validation,
                                                                                              Y_training, model_parameters,
                                                                                              verbose)
                if model_type == 'regression':
                    validation_predictions, train_predictions, trained_model = GBM_REGRESSOR(X_training, X_validation,
                                                                                             Y_training, model_parameters,
                                                                                             verbose)
            elif model == 'glm' or model == 'mixed_glm':

                if model_type == 'classification':
                    validation_predictions, train_predictions, trained_model = GLM_CLASSIFIER(X_training, X_validation,
                                                                                              Y_training, model_parameters,
                                                                                              verbose)
                if model_type == 'regression':
                    validation_predictions, train_predictions, trained_model = GLM_REGRESSOR(X_training, X_validation,
                                                                                             Y_training, model_parameters,
                                                                                             verbose)
            elif model == 'knn' or model == 'mixed_knn':

                if model_type == 'classification':
                    validation_predictions, train_predictions, trained_model = KNN_CLASSIFIER(X_training, X_validation,
                                                                                              Y_training, model_parameters,
                                                                                              verbose)
                if model_type == 'regression':
                    validation_predictions, train_predictions, trained_model = KNN_REGRESSOR(X_training, X_validation,
                                                                                             Y_training, model_parameters,
                                                                                             verbose)
            elif model == 'nn' or model == 'mixed_nn':

                if model_type == 'classification':
                    validation_predictions, train_predictions, trained_model = NN_CLASSIFIER(X_training, X_validation,
                                                                                              Y_training, model_parameters,
                                                                                              verbose)
                if model_type == 'regression':
                    validation_predictions, train_predictions, trained_model = NN_REGRESSOR(X_training, X_validation,
                                                                                            Y_training, model_parameters,
                                                                                            verbose)
        except Exception as ex:
            raise Exception("{0} model\n\t   {1}\n\n{2}".format(model.upper(),str(ex),traceback.format_exc()))
            
        if model_type == 'classification':
            if (model == 'nn') and (not np.allclose(1, train_predictions.sum(axis=1))) or (not np.allclose(1, validation_predictions.sum(axis=1))):
                 raise Exception(
                     "The output predictions of the neural network model need to be probabilities "
                     "i.e. they should sum up to 1.0 over classes. But the output does not match the condition. "
                     "Revise the model parameters to solve the problem.")
        
    elif callable(model):
        try:
            train_predictions, validation_predictions, trained_model = model(X_training, X_validation, Y_training)
        except Exception as ex:
            raise Exception("The user-defined model '{0}' encountered an error:\n\t   {1}\n\n{2}".format(model.__name__,str(ex),traceback.format_exc()))
        
        if (type(train_predictions) not in (np.ndarray,list)) or (type(validation_predictions) not in (np.ndarray,list)):
            raise Exception("The output predictions of the user-defined model must be of type array.")
        
        train_predictions = np.array(train_predictions)
        validation_predictions = np.array(validation_predictions)
        
        if (len(train_predictions) != len(X_training)) or (len(validation_predictions) != len(X_validation)):
            raise Exception("The output of the user-defined model has a different length from the input data.")
        
        if model_type == 'classification':
            try:
                train_predictions.shape[1]
                validation_predictions.shape[1]
            except IndexError:
                raise Exception("The output of the user_defined classification model must be an array of shape (n_samples,n_classes).")
                                    
            if ((train_predictions.shape[1] != len(present_labels)) or (validation_predictions.shape[1] != len(present_labels))):
                raise Exception("The probability predictions of the user-defined model are not compatible with the number of classes in the input data.")
            
            if (not np.allclose(1, train_predictions.sum(axis=1))) or (not np.allclose(1, validation_predictions.sum(axis=1))):
                 raise Exception(
                     "The output predictions of the user-defined model need to be probabilities "
                     "i.e. they should sum up to 1.0 over classes")

    else:
        raise ValueError(
            "The model must be name of one of the supported models: {'gbm', 'glm', 'knn', 'nn'} Or user defined function.")
    
    # adding the zero probability for the labels which are not included in the train data and thus are 
    # not considered in the predictions
    if (model_type == 'classification') and (labels is not None):
        
        train_predictions = complete_predicted_probabilities(predictions = train_predictions,
                                                             all_labels = labels, present_labels = present_labels)
        validation_predictions = complete_predicted_probabilities(predictions = validation_predictions,
                                                                  all_labels = labels, present_labels = present_labels)
        
    return train_predictions, validation_predictions, trained_model


def train_evaluate(training_data, validation_data, model, model_type, model_parameters = None,
                   labels = None, base_models = None, verbose = 0):
    
    
    # initializing
    train_predictions = None
    validation_predictions = None
    trained_model = None
    base_model_list = []
    base_model_name_list = []
    base_model_parameter_list = []
    supported_models_name = ['gbm', 'glm', 'knn', 'nn']
    
    
    if type(training_data) == str:
        try:
            training_data = pd.read_csv(training_data)
        except FileNotFoundError:
            raise FileNotFoundError("File '{0}' does not exist.".format(training_data))
    elif type(training_data) != pd.DataFrame:
      raise TypeError("The training_data input must be a data frame or data address.")

    if type(validation_data) == str:
        try:
            validation_data = pd.read_csv(validation_data)
        except FileNotFoundError:
            raise FileNotFoundError("File '{0}' does not exist.".format(validation_data))
    elif type(validation_data) != pd.DataFrame:
      raise TypeError("The validation_data input must be a data frame or data address.")

    # split features and target
    if 'spatial id' in training_data.columns.values or 'temporal id' in training_data.columns.values:
        X_training = training_data.drop(['Target', 'spatial id', 'temporal id'], axis=1)
    else:
        X_training = training_data.drop(['Target'], axis=1)
    Y_training = np.array(training_data['Target']).reshape(-1)
    if 'spatial id' in validation_data.columns.values or 'temporal id' in validation_data.columns.values:
        X_validation = validation_data.drop(['Target', 'spatial id', 'temporal id'], axis=1)
    else:
        X_validation = validation_data.drop(['Target'], axis=1)
    Y_validation = np.array(validation_data['Target']).reshape(-1)

    if 'Normal target' in X_training.columns:
        X_training = X_training.drop(['Normal target'], axis=1)
        X_validation = X_validation.drop(['Normal target'], axis=1)
        
    # check base model input
    if base_models is not None:
        if type(base_models) not in (np.ndarray, list):
            raise TypeError('The base_models must be of type list.')
            
        for item_number, item in enumerate(base_models):

            # if the item is the dictionary of model name and its parameters
            if type(item) == dict:       
                base_model = list(item.keys())[0]

                # if the dictionary contain only one of the supported models
                if (len(item) == 1) and (base_model in supported_models_name):

                    base_model_list.append(base_model)
                    # if model is not duplicate 
                    if base_model not in base_model_name_list:
                        base_model_name_list.append(base_model)
                    else:
                        base_model_name_list.append(base_model + str(item_number))

                    # if the value of the model name is dictionary of models parameter list
                    if type(item[base_model]) == dict:
                        base_model_parameter_list.append(item[base_model])
                    else:
                        base_model_parameter_list.append(None)
                        print("\nWarning: The values in the dictionary items of base_models list must be a dictionary of the model hyper parameter names and values. Other values will be ignored.\n")
                else:
                    print("\nWarning: Each dictionary item in base_models list must contain only one item with a name of one of the supported models as a key and the parameters of that model as value. The incompatible cases will be ignored.\n")

            # if the item is only name of model whithout parameters
            elif type(item) == str:
                if (item in supported_models_name):
                    base_model_list.append(item)
                    base_model_parameter_list.append(None)
                    if (item not in base_model_name_list):
                        base_model_name_list.append(item)
                    else:
                        base_model_name_list.append(item + str(item_number))
                else:
                    print("\nWarning: The string items in the base_models list must be one of the supported model names. The incompatible cases will be ignored.\n")

            # if the item is user defined function
            elif callable(item):
                if item.__name__ in supported_models_name:
                    raise Exception("User-defined model names must be different from predefined models:['knn', 'glm', 'gbm', 'nn']")
                base_model_list.append(item)
                base_model_name_list.append(item.__name__)
                base_model_parameter_list.append(None)

            else:
                print("\nWarning: The items in the base_models list must be of type string, dict or callable. The incompatible cases will be ignored.\n")

        if len(base_model_list) < 1:
            raise ValueError("There is no item in the base_models list or the items are invalid.")
        
    # if model is non-mixed
    if base_models is None:
        train_predictions, validation_predictions, trained_model = run_model(X_training, X_validation, Y_training, model, model_type,\
                                                                                supported_models_name,model_parameters, labels, verbose)
    # if model is mixed
    else:
        mixed_X_training = pd.DataFrame()
        mixed_X_validation = pd.DataFrame()

        for base_model_number, base_model_name in enumerate(base_model_name_list):
            train_predictions, validation_predictions, trained_model = run_model(X_training.copy(), X_validation.copy(), Y_training.copy(),\
                                                                                 base_model_list[base_model_number], model_type,\
                                                                                  supported_models_name, base_model_parameter_list[base_model_number],\
                                                                                 labels, verbose)
            if model_type == 'classification':
#                 train_predictions = [labels[index] for index in np.argmax(train_predictions, axis=1)]
#                 validation_predictions = [labels[index] for index in np.argmax(validation_predictions, axis=1)]

                total_class_number = 1 if train_predictions.shape[1] == 2 else train_predictions.shape[1]
                for class_num in range(total_class_number):
                    mixed_X_training[base_model_name+str(class_num)] = list(train_predictions[:,class_num])
                    mixed_X_validation[base_model_name+str(class_num)] = list(validation_predictions[:,class_num])
            else:
                mixed_X_training[base_model_name] = list(train_predictions)
                mixed_X_validation[base_model_name] = list(validation_predictions)

        train_predictions, validation_predictions, trained_model = run_model(mixed_X_training, mixed_X_validation, Y_training, model, model_type,\
                                                                             supported_models_name, model_parameters, labels, verbose)
    
    return train_predictions, validation_predictions, trained_model


#####################################################################################################

def inner_train_evaluate(training_data, validation_data, model, model_type, model_parameters = None, labels = None, base_models=None, verbose = 0):
    
    train_predictions, validation_predictions, trained_model = train_evaluate(training_data = training_data,
                                                                              validation_data = validation_data,
                                                                              model = model, 
                                                                              model_type = model_type,
                                                                              model_parameters = model_parameters, 
                                                                              labels = labels,
                                                                              base_models = base_models,
                                                                              verbose = verbose)
    
    if model == 'gbm' or model == 'mixed_gbm':
        # get the number of trees
        number_of_parameters = None # trained_model.n_estimators_
        
    elif model == 'glm' or model == 'mixed_glm':
        # get the number of coefficients and intercept
        if model_type == 'classification':
            number_of_parameters = trained_model.coef_.shape[0]*trained_model.coef_.shape[1]
            if not all(trained_model.intercept_ == 0):
                number_of_parameters += trained_model.intercept_.shape[0]
        if model_type == 'regression':
            number_of_parameters = None # trained_model.coef_.shape[0]
            # if trained_model.get_params()['fit_intercept']:
                # number_of_parameters += 1
                
    elif model == 'knn' or model == 'mixed_knn':
        # get the number of nearest neighbours
        number_of_parameters = None # trained_model.get_params()['n_neighbors']
        
    elif model == 'nn' or model == 'mixed_nn':
        # get the number of parameters
        number_of_parameters = None # trained_model.count_params()
        
    else:
        number_of_parameters = None
        
    return train_predictions, validation_predictions, trained_model, number_of_parameters
