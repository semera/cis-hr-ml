# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
import numpy;

#cd s:\semera\OneDrive\Dokumenty\Anaconda\hr_cis\priklad1 

# learning dataset
dataset = numpy.loadtxt("data/learning.csv", delimiter=",")
X = dataset[:,0:3]
Y = dataset[:,3]

# -----------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# ----------------------------------------------------------------------------- 

# defaultni model pro KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=3, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
 
# -----------------------------------------------------------------------------    

seed = 7
numpy.random.seed(seed)

# model
model = KerasClassifier(build_fn=create_model, verbose=2)

# vyber testu
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [5, 20];# 50, 100, 150]
batches =[500, 2000]; #  [5, 10, 20]

# kalkulace
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

# vysledky
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param)) 

"""  
Best: 0.908960 using {'batch_size': 5, 'epochs': 5, 'init': 'uniform', 'optimizer': 'rmsprop'}
0.849800 (0.031419) with: {'batch_size': 5, 'epochs': 5, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.859200 (0.010833) with: {'batch_size': 5, 'epochs': 5, 'init': 'glorot_uniform', 'optimizer': 'adam'}
0.866060 (0.034252) with: {'batch_size': 5, 'epochs': 5, 'init': 'normal', 'optimizer': 'rmsprop'}
0.894740 (0.010143) with: {'batch_size': 5, 'epochs': 5, 'init': 'normal', 'optimizer': 'adam'}
0.908960 (0.010848) with: {'batch_size': 5, 'epochs': 5, 'init': 'uniform', 'optimizer': 'rmsprop'}
0.832840 (0.051518) with: {'batch_size': 5, 'epochs': 5, 'init': 'uniform', 'optimizer': 'adam'}
0.782480 (0.052361) with: {'batch_size': 10, 'epochs': 5, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.781440 (0.031689) with: {'batch_size': 10, 'epochs': 5, 'init': 'glorot_uniform', 'optimizer': 'adam'}
0.833680 (0.079770) with: {'batch_size': 10, 'epochs': 5, 'init': 'normal', 'optimizer': 'rmsprop'}
0.906200 (0.003888) with: {'batch_size': 10, 'epochs': 5, 'init': 'normal', 'optimizer': 'adam'}
0.892840 (0.018610) with: {'batch_size': 10, 'epochs': 5, 'init': 'uniform', 'optimizer': 'rmsprop'}
0.853980 (0.039086) with: {'batch_size': 10, 'epochs': 5, 'init': 'uniform', 'optimizer': 'adam'}
0.717500 (0.029244) with: {'batch_size': 20, 'epochs': 5, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.658440 (0.113328) with: {'batch_size': 20, 'epochs': 5, 'init': 'glorot_uniform', 'optimizer': 'adam'}
0.882240 (0.014787) with: {'batch_size': 20, 'epochs': 5, 'init': 'normal', 'optimizer': 'rmsprop'}
0.902020 (0.013962) with: {'batch_size': 20, 'epochs': 5, 'init': 'normal', 'optimizer': 'adam'}
0.880620 (0.017773) with: {'batch_size': 20, 'epochs': 5, 'init': 'uniform', 'optimizer': 'rmsprop'}
0.852720 (0.055801) with: {'batch_size': 20, 'epochs': 5, 'init': 'uniform', 'optimizer': 'adam'}
"""