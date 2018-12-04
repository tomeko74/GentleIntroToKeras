from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 9
numpy.random.seed(seed)

from pandas import read_csv
filename = 'BBCN.csv'
dataframe = read_csv(filename)

array = dataframe.values

X = array[:,0:11]
Y = array[:,11]

dataframe.head()

model = Sequential()
model.add(Dense(16, input_dim=11, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=300, batch_size=50)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
