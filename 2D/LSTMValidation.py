from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

x = [[[(i + j)/1000] for i in range(30)] for j in range(100)]
y = [(i + 5)/1000 for i in range(100)]

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=4)

model = Sequential()
model.add(LSTM(5, batch_input_shape=(None, 30, 1), return_sequences=True))
model.add(LSTM(5, return_sequences=False))
model.add(Dense(1, activation="relu"))
model.compile(loss="MSE", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=800, validation_split=0.2, shuffle=True, verbose=2)

results = model.predict(x_test)
plt.scatter(range(20), results, c="r")
plt.scatter(range(20), y_test, c="g")
plt.show()
plt.plot(history.history["loss"])
plt.show()
