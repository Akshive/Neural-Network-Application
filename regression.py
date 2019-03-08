'''
we will use boston_housing dataset
the steps are:
   import data
   prepare data using feature normalisation because range of different features are different
   build the network
   train model, use k-fold validation(useful for small datasets)
   check performance by Mean Absolute Error
'''


import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import boston_housing

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
    
    
def smooth_points(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
  
    

(train_data, train_label), (test_data, test_label) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std_dev = train_data.std(axis=0)
train_data /= std_dev

test_data -= mean
test_data /= std_dev

    
k = 4
num_val_samples = len(train_data)//k
num_epochs = 80
all_mae_histories = []

for i in range(k):
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_label=train_label[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data=np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_label=np.concatenate([train_label[:i*num_val_samples], train_label[(i+1)*num_val_samples:]],axis=0)
    model = build_model()
    history=model.fit(partial_train_data, partial_train_label, validation_data=(val_data, val_label), epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
  

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


smooth_mae_history = smooth_points(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

test_mse_score, test_mae_score = model.evaluate(test_data, test_label)
print(test_mae_score)
