NON_FEATURE_COLUMNS_NAMES = ['spatial id', 'temporal id']

TARGET_COLUMN_NAME = 'Target'

NORMAL_TARGET_COLUMN_NAME = 'Normal target'

BASIC_TARGET_COLUMN_NAMES = ['Target (normal)',
                             'Target (augmented on z units - normal)',
                             'Target (differential)',
                             'Target (augmented on z units - differential)',
                             'Target (moving average on x units)',
                             'Target (augmented on z units - moving average on x units)',
                             'Target (cumulative)',
                             'Target (augmented on z units - cumulative)']

TEST_TYPES = ['one-by-one', 'whole-as-one']

TARGET_MODES = ['normal', 'cumulative', 'differential', 'moving_average']

RANKING_METHODS = ['mRMR', 'correlation', 'variance']

FEATURE_SELECTION_TYPES = ['covariate', 'feature']

PRE_DEFINED_MODELS = ['nn', 'knn', 'glm', 'gbm']

MODEL_TYPES = ['regression', 'classification']

PERFORMANCE_MEASURES = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score', 'AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']

REGRESSION_PERFORMANCE_MEASURES = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']

CLASSIFICATION_PERFORMANCE_MEASURES = ['AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']

PERFORMANCE_BENCHMARKS = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score', 'AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']

REGRESSION_PERFORMANCE_BENCHMARKS = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']

CLASSIFICATION_PERFORMANCE_BENCHMARKS = ['AIC', 'BIC', 'likelihood', 'AUC', 'AUPR']

PERFORMANCE_MODES_STARTS_WITH = ['normal', 'cumulative', 'moving_average']

FEATURE_SCALERS = ['logarithmic', 'normalize', 'standardize', None]

TARGET_SCALERS = ['logarithmic', 'normalize', 'standardize', None]

SPLITTING_TYPES = ['training-validation', 'cross-validation']

VERBOSE_OPTIONS = [0, 1, 2]

SCENARIOS = ['max', 'min', 'mean', 'current', None]
