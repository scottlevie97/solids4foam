import numpy as np
import pandas as pd

import os
# 2 = INFO and WARNING messages are not printed
# Stop tensorflow writing messages about improving performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

import timeit

from settings import *

start = timeit.default_timer()

def inputsoutputsAllfiles(trainIterations, filepatharray, Timestepsize, maxtimestep):

    def xyz(iteration, filepath, Timestepsize, Timestep):

        Time = Timestep*Timestepsize 
        if Time%1 == 0:
            Time =int(Time)
        else: 
            Time = round(Time,2)
        my_file = open(filepath + str(Time) +"/solidsCellD_iteration"+str(iteration))
        string_list = my_file.readlines()
        index=18
        string_list = string_list[index+2:len(string_list)] 
        string_list = string_list[0:len(string_list)-4] 
        i=0
        for i in range(len(string_list)):
            string_list[i]=string_list[i].replace("(","")
            string_list[i]=string_list[i].replace(")","")
            string_list[i]=string_list[i].split()
        arr=np.array(string_list)
        arr= arr.astype(np.float64)
        x = arr[:,0]
        y = arr[:, 1]
        z = arr[:, 2]

        return x, y, z #, arr
        
      
    def maxIteration (Timestep, Timestepsize, filepath):
        Time = Timestep*Timestepsize
        Time = round(Time,2)   # Rounds to deicimal place
        file = filepath + "residual.dat"

        df = pd.read_csv(file)

        df.columns = ['One'] #Labels first Column

        df['One'] = df['One'].astype('string') # Converts to string

        new = df['One'].str.split(' ', expand = True) #splits into new dataframe on spaces

        df['TimeStep']= new[0].astype('float')  # labels new columns and inserts in dataframe
        df['solverPerfInitRes']= new[1].astype('float')  # labels new columns and inserts in dataframe
        df['residualvf']= new[2].astype('float')  # labels new columns and inserts in dataframe
        df['materialResidual']= new[3].astype('float')  # labels new columns and inserts in dataframe
        df["IterationNumber"]= new[4].astype('float') # labels new columns and inserts in dataframe

        df.drop(columns =['One'], inplace = True) #Removes original column

        array = np.array
        array = []

        for i in range(len(df)):        #loops through residual file
            if df['TimeStep'][i] == Time:   # Finds values for a timestep
                x = df['IterationNumber'][i]
                array.append(x)            # Creates arrays of iteration numbers
                
        max = np.max(array) # returns max of array
        max=int(max)
        return max
    
    def inputsoutputsAllTimesteps(trainIterations, filepath, Timestepsize):

        def inputsoutputsTimestep(trainIterations, filepath, Timestepsize, Timestep):

            max = maxIteration(Timestep, Timestepsize, filepath)

            x,y,z = xyz(1, filepath, Timestepsize, Timestep)
            matrix = np.array(np.vstack((x,y,z))) #Gives matrix a starting value which is removed later

            for i in range(0,trainIterations): # loops for each iteration up to trainIteration
                x,y,z = xyz(i,filepath, Timestepsize, Timestep)
                arr = np.vstack((x,y,z))  # gives array with x, y, z values for each iteration
                matrix = np.concatenate((matrix, arr))   # combines arrays

            inputs = matrix[3:,:] # removes initial value

            xmax, ymax, zmax = xyz(max, filepath, Timestepsize, Timestep)
            outputs = np.vstack((xmax,ymax,zmax)) # array of values at max iteration

            inputs = np.transpose(inputs)  # multiple columns : (x1, y1, z1, x2, y2, z2 ...)
            outputs = np.transpose(outputs)

            return inputs, outputs

        arrinputs = np.ones((1, trainIterations*3)) # first row to be removed after
        arroutputs = np.ones((1, 3))

        #combines inputs and outputs of all timesteps

        for Timestep in range(1,maxtimestep+1):
            inputs, outputs = inputsoutputsTimestep(trainIterations, filepath, Timestepsize, Timestep)
            arrinputs = np.vstack((arrinputs,inputs))
            arroutputs = np.vstack((arroutputs,outputs))

        arrinputs = arrinputs[1:,:] #removes oriinal row of ones
        arroutputs = arroutputs[1:,:]#removes oriinal row of ones

        outputs = arroutputs
        inputs = arrinputs

        return inputs, outputs   # Returns the inputs and outputs of All timesteps for one file

    arrinputs = np.ones((1, trainIterations*3))
    arroutputs = np.ones((1, 3))

    for i in range(0,len(filepatharray),1):
        inputs, outputs = inputsoutputsAllTimesteps(trainIterations, filepatharray[i], Timestepsize)
        arrinputs = np.vstack((arrinputs,inputs))
        arroutputs = np.vstack((arroutputs,outputs))

    arrinputs = arrinputs[1:,:] #removes oriinal row of ones
    arroutputs = arroutputs[1:,:]#removes oriinal row of ones

    outputs = arroutputs
    inputs = arrinputs

    return inputs, outputs

# Print settings

print("\n ML Settings: \n")

print("filepath1: " + filepath1 + "\n")
print("trainIterations: " + str(trainIterations) + "\n")
print("Timestepsize: " + str(Timestepsize) + "\n")
print("maxtimestep: " + str(maxtimestep) + "\n")


filepatharray = [filepath1]

inputs, outputs = inputsoutputsAllfiles(trainIterations, filepatharray, Timestepsize, maxtimestep)# define model

X = inputs
y = outputs

scalerI = StandardScaler().fit(X) #FitScaler
scalerO = StandardScaler().fit(y)

import pickle
pickle.dump(scalerI, open('ML_model/scalerI.pkl','wb'))
pickle.dump(scalerO, open('ML_model/scalerO.pkl','wb'))

scaledX = scalerI.transform(X) 
scaledY = scalerO.transform(y)


if answer_trainOrLoadModel == ("t"):

    model = Sequential()
    model.add(Dense(200, input_dim=len(X[0,:]), 
                    activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(150, 
                    activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(50, 
                    activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit model
    # try without verbose for speed
    history = model.fit(scaledX, scaledY, epochs=answer_epochs, verbose=verbose_setting)


    # evaluate the model
    train_mse = model.evaluate(scaledX, scaledY, verbose=0)


    print('Train MSE: %f' % (train_mse))

    # plot loss during training
    plt.title('Loss / Mean Squared Error')
    plt.plot(history.history['loss'], label='train')
    plt.yscale('log')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    if trainingGraph: plt.show()

    model.save('ML_model/ANN_model')

elif answer_trainOrLoadModel == ("l"):
    model = keras.models.load_model('ML_model/ANN_model')


scaledPrediction = model.predict(scaledX)
prediction = scalerO.inverse_transform(scaledPrediction)

# Save Model
model.save('ML_model/fdeepTrial.h5', include_optimizer=False)

stop = timeit.default_timer()

print('Run time: ', stop - start)

