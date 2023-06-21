from numpy import array
from keras import *

data=[
        [18,2,0,0,2100],
        [32,4,10,0,3400],
        [44,2,20,0,3200],
        [60,6,343,0,550],
        [20,2,34,0,2300],
        [24,4,0,1,2700],
        [44,5,21,1,3200],
        [52,2,32,1,3200]
        ]

data_array=array[data]

model=Sequential()
model.add(layers.Dense(units=16,input_shape=[4]))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=64))
model.add(layers.Dense(units=1))
entree=data_array[0:8,0:4]
sortie=data_array[0:8,4]
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x=entree,y=sortie,epochs=5000)
gabriel=array([[18,1,1,0]])
louise=array([[52,7,20,0]])
print(model.predict(gabriel))
print(model.predict(louise))
