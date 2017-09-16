from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Activation
#from keras.layers import Dropout
#from keras.layers import LSTM
import numpy

# cd S:\semera\OneDrive\Dokumenty\Anaconda\hr_cis\priklad1

# learning dataset
dataset = numpy.loadtxt("data/learning.csv", delimiter=",")
X = dataset[:,0:3]
Y = dataset[:,3]

# test dataset, ten splitovat nemusim
preX = numpy.loadtxt("data/test1000.csv", delimiter=",") 

# -----------------------------------------------------------------------------

# stupidni rozdil mezi tensorflow a theano?
model = Sequential()
model.add(Dense(50, input_dim=3, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # nic
#model.fit(X, Y, epochs=2000, batch_size=100, verbose=2) # 999 final model model
model.fit(X, Y, epochs=10, batch_size=5, verbose=1); # 998 predvadeci verze 

# -----------------------------------------------------------------------------

# ulozeni nebo nacteni modelu
model.load_weights('models/s999.weights.h5')
#model.save_weights('models/s999.weights.h5')

# -----------------------------------------------------------------------------

# predikce
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# -----------------------------------------------------------------------------

# vysledky
predictions = model.predict(preX)
for p in predictions:
    print(int(round(p[0], 0)))
    
# -----------------------------------------------------------------------------

# ukazka histogramu    
import matplotlib.pyplot as plt
import pandas
url = "data/learning.csv"
names = ['typ', 'O1', 'O2', 'result']
data = pandas.read_csv(url, names=names)
data.hist()
plt.show()
    