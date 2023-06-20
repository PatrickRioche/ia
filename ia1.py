from keras import *
model=Sequential()
model.add(layers.Dense(units=3,input_shape=[1]))
model.add(layers.Dense(units=2))
model.add(layers.Dense(units=1))
entree=[1,2,3,4,5]
sortie=[3,6,9,12,15]
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x=entree,y=sortie,epochs=5000)
print(model.predict([6]))
print(model.predict([7]))
print(model.predict([8]))
