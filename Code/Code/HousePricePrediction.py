import pandas as pd
import numpy as np
import keras as k
import tensorflow 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Model, load_model
from pickle import dump
import math
from tensorflow.python.client import device_lib 
from matplotlib import rcParams, cycler


print(device_lib.list_local_devices())

BATCH_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT = 0.004
VALIDATION_SPLIT = 0.1
OPTIMIZER = 'adam'

LAYER_1 = 900
LAYER_2 = 150
LAYER_3 = 700
LAYER_4 = 550
LAYER_5 = 950
LAYER_6 = 950

def create():
    model = tensorflow.keras.Sequential()    
    model.add(tensorflow.keras.layers.Dense(128, input_dim=len(getKeys) - 1, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(1))

    def RMSE(yAxisOriginal, yAxisPrediction):
        yAxisPrediLog = k.log(k.clip(yAxisPrediction, k.epsilon(), None) + 1.)
        yAxisOrigLog = k.log(k.clip(yAxisOriginal, k.epsilon(), None) + 1.)
        return k.sqrt(k.mean(k.square(yAxisPrediLog - yAxisOrigLog), axis = -1))

    model.compile(loss='mean_squared_error', 
                  optimizer=tensorflow.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
                 )
    
    return model

readData = pd.read_csv("HousePrice.csv")

# Get keys
getKeys = readData.keys()

# Target key
targetKey = 'deptFreePrice'

# Initialize scaler
minMaxScaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
trainedScaler = minMaxScaler.fit_transform(readData[getKeys])

print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(minMaxScaler.scale_[6], minMaxScaler.min_[6]))

# Store the values to scale price back to understandable form
scalePrice = minMaxScaler.scale_[6]
addScalePRice = minMaxScaler.min_[6]

scalerData = {'scale_': minMaxScaler.scale_, 'min_': minMaxScaler.min_}
scalerDataFrame = pd.DataFrame(scalerData, columns = ['scale_', 'min_'])


scaledTrain = pd.DataFrame(trainedScaler, columns=getKeys)

# Divide dataset into training and testing sets
trainAndTest = np.random.rand(len(scaledTrain)) < 0.8

xAxis, yAxis = scaledTrain.drop(targetKey, axis=1).values, scaledTrain[targetKey].values
XAxisTrainedData =scaledTrain[trainAndTest].values

xAxisTrain, xAxisTest = scaledTrain[trainAndTest].drop(targetKey, axis=1).values, scaledTrain[trainAndTest].drop(targetKey, axis=1).values
yAxisTrain, yAxisTest = scaledTrain[trainAndTest][targetKey].values, scaledTrain[trainAndTest][targetKey].values

print(xAxisTrain)
print(yAxisTrain)
print("The length of X axis Train and Test Data is ",len(xAxis))
print("The length of Y axis Train and Test Data is ",len(yAxis))
print("The length of X axis Train Data is ",len(xAxisTrain))
print("The length of X axis Test Data is ",len(xAxisTest))

earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
modelCheckPoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.90,
                              patience=10, min_lr=0.000001)
number = 10
modelFit = []
model = create()
print("Modesl Summary",model.summary())
modelFit = model.fit(
    xAxisTrain,
    yAxisTrain,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=2000,
    shuffle=True,
    verbose=1,
    callbacks=[earlyStopping, modelCheckPoint, reduceLROnPlateau
               #tensorboard_callback
              ]
)

saveBestModel= model.save('my_model')
saveBestModel = tensorflow.keras.models.load_model('my_model', compile=False)
#model = tf.keras.models.load_model('my_model', compile=False)
bestModel = tensorflow.keras.models.load_model('my_model', compile=False)
bestModel_2 = tensorflow.keras.models.load_model('my_model', compile=False)
bestModel_3 = tensorflow.keras.models.load_model('my_model', compile=False)
bestModel_4 = tensorflow.keras.models.load_model('my_model', compile=False)
bestModel_5 = tensorflow.keras.models.load_model('my_model', compile=False)

figure, axis = plt.subplots(figsize=(20, 10))
loss = axis.plot(modelFit.history['loss'], label='Loss')
valLoss = axis.plot(modelFit.history['val_loss'], label='val_loss')
axis.legend(loc='upper right',fontsize=15)

bestModel.compile(OPTIMIZER,loss = 'mean_squared_error', metrics=['accuracy'])

testErrorRate = bestModel.evaluate(xAxisTrain, yAxisTrain, verbose=0)
#print("MSE for the data set is: {}".format(test_error_rate))
getPredictors = getKeys.drop(targetKey)

# Make predictions from test set
getPrediction = bestModel.predict(xAxisTest)

MAEValue = 0
RMEValue = 0
RMSEValue = 0
predValues = []
orgValues = []

# Scale target values back to normal
for test in yAxisTest:
    axisTest = test
    axisTest -= addScalePRice
    axisTest /= scalePrice
    orgValues.append(axisTest)
    
orgValues = np.asarray(orgValues)

# Scale predicted values back to normal
for prediction in getPrediction:
    predicted = prediction
    predicted -= addScalePRice
    predicted /= scalePrice
    predValues.append(predicted)
    
# Calculate metrics for sensitivity analysis
for i in range(len(predValues)):
    MAEValue += abs(predValues[i] - orgValues[i])
    RMEValue += abs(predValues[i] - orgValues[i]) / orgValues[i] * 100
    RMSEValue += (predValues[i] - orgValues[i])**2
    

MAEValue = float(MAEValue / len(predValues))
RMEValue = float(RMEValue / len(predValues))
RMSEValue = math.sqrt(RMSEValue / len(predValues))
print("The MAE Value is ",MAEValue)
print("The RME Value is ",RMEValue)
print("The RMSE Value is ",RMSEValue)
newValues = np.asarray(predValues)
#print("\nRMSPE for the data set is: {0:.2f}%".format(error_p))

print("Original Values are",orgValues.shape)
print("Predicted Values are",newValues.shape)

# Calculate r squared
orgValues = np.array([orgValues],order='C')
orgValues.resize((817,))
orgValues.shape
newValues = np.array([newValues],order='C')
newValues.resize((817,))
newValues.shape
correlationMatrix = np.corrcoef(orgValues, newValues)
correlationXY = correlationMatrix[0,1]
rSquaredValue = correlationXY**2

print("The R Squared Value is",rSquaredValue)
print("Min value -", min(newValues), " Min original - ", min(orgValues))
print("Max value -", max(newValues), " Max original - ", max(orgValues))
print("Mean -", np.nanmean(newValues), " Mean original - ", np.nanmean(orgValues))
print("Std -", np.nanstd(newValues), " Std original - ", np.nanstd(orgValues))

error_rate = bestModel.evaluate(xAxisTest, yAxisTest, verbose=0)
#print("DIF: ", (error_rate - test_error_rate))
#print("\nMSE for the data set is: {0:.4f}".format(error_rate))
print("MSE for the data set is ",error_rate)
map = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=map(np.linspace(0, 1, 2)))

figure1, axis1 = plt.subplots(figsize=(15, 15))
lines = axis1.plot(orgValues, orgValues, label='Predicted', linewidth=2)
axis1.scatter(orgValues, newValues, color='r')
axis1.set_xlabel('Original price',fontsize=18)
axis1.set_ylabel('Predicted price',fontsize=18)

text = '\n'.join(("MAE: {0:.0f} £".format(MAEValue), "RME: {0:.2f}%".format(RMEValue), "RMSE: {0:.0f} £".format(RMSEValue), "R²: {0:.2f}".format(rSquaredValue)))
axis1.text(0.01, 0.91, text, fontsize=16,
               verticalalignment='bottom', transform=axis1.transAxes)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


rooms = []
predSeperately = [[], [], [], [], [], []]
orgSeperately= [[], [], [], [], [], []]
result = []
for i in range(len(xAxisTest)):
    scaled = math.floor((xAxisTest[i][3] - minMaxScaler.min_[3]) / minMaxScaler.scale_[3])
    rooms.append(scaled)
    
for i in range(len(newValues)):
    if rooms[i] == 1:
        predSeperately[0].append(newValues[i])
        orgSeperately[0].append(orgValues[i])
    elif rooms[i] == 2:
        predSeperately[1].append(newValues[i])
        orgSeperately[1].append(orgValues[i])
    elif rooms[i] == 3:
        predSeperately[2].append(newValues[i])
        orgSeperately[2].append(orgValues[i])
    elif rooms[i] == 4:
        predSeperately[3].append(newValues[i])
        orgSeperately[3].append(orgValues[i])
    elif rooms[i] == 5:
        predSeperately[4].append(newValues[i])
        orgSeperately[4].append(orgValues[i])
    elif rooms[i] == 6:
        predSeperately[5].append(newValues[i])
        orgSeperately[5].append(orgValues[i])
      

for n in range(len(predSeperately)):
    for i in range(len(predSeperately[n])):
        MAEValue += abs(predSeperately[n][i] - orgSeperately[n][i])
        RMEValue += abs(predSeperately[n][i] / orgSeperately[n][i] - 1) * 100
        RMSEValue += (predSeperately[n][i] - orgSeperately[n][i])**2

    correlation_matrix = np.corrcoef(orgSeperately[n], predSeperately[n])
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    MAEValue = MAEValue / len(predSeperately[n])
    RMEValue = RMEValue / len(predSeperately[n])
    RMSEValue = math.sqrt(RMSEValue / len(predSeperately[n]))
    result.append([MAEValue,RMEValue,RMSEValue,r_squared])
    MAEValue, RMEValue, RMSEValue = 0, 0, 0

print("The Final Predicted Values are",result)


