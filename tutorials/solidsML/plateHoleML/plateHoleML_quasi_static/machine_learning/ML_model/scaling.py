import numpy as np
import pandas as pd
import pickle

import sys
sys.path.insert(0, '../')
from settings import *

import os
# 2 = INFO and WARNING messages are not printed
# Stop tensorflow writing messages about improving performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
scalerI = pickle.load(open('scalerI.pkl','rb'))
scalerO = pickle.load(open('scalerO.pkl','rb'))

mean = np.hstack((scalerI.mean_, scalerO.mean_))
std = np.hstack((scalerI.scale_, scalerO.scale_))

names = np.array([])
for i in np.arange(1,len(mean)/3):

    names = np.append(
                        names, 
                        [
                            "X"+str(int(i)),
                            "Y"+str(int(i)),
                            "Z"+str(int(i))
                        ]      
                    )
names = np.append(
                    names,
                    [
                        "X_ouptut",
                        "Y_output",
                        "Z_output"
                    ]
                )

df = pd.DataFrame(data = np.vstack((mean, std)), columns = names, index = ["mean", "std"])

df

df.to_csv('scalingData.csv')

answer_solidsProp = "y"

#  Create mean and std strings
mean_string = str()
for i in np.arange(0,len(mean)/3):

    mean_string =  mean_string + "\t\t(" + str(mean[3*int(i)]) + "\t\t"+ str(mean[3*int(i) + 1]) + "\t\t"+ str(mean[3*int(i) + 2])+ ")\n"

std_string = str()
for i in np.arange(0,len(std)/3):

    std_string =  std_string + "\t\t(" + str(std[3*int(i)]) + "\t\t"+ str(std[3*int(i) + 1]) + "\t\t"+ str(std[3*int(i) + 2])+ ")\n"

# Write these to solidProperties file

def stringWrite (filename, string_find, string_insert):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    index = 0
    for line in lines:
        index += 1 

        if string_find in line:
            break

    lines[index + 1] = string_insert
    f = open(filename, "w")
    f.writelines(lines)
    f.close()

filename = "solidProperties"

string_find1 = 'kerasScalingMeans'
string_insert1 = mean_string

string_find2 = 'kerasScalingStds'
string_insert2 = std_string

if answer_solidsProp == "y":

    stringWrite(filename, string_find1, string_insert1)
    stringWrite(filename, string_find2, string_insert2)

# Write training iterations

def stringWrite2 (filename, string_find, string_insert):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    index = 0
    for line in lines:
        index += 1 

        if string_find in line:
            break

    lines[index -1] = "\t" + string_find1 + " " +  string_insert1 + "; \n" 
    
    f = open(filename, "w")
    f.writelines(lines)
    f.close()

filename = "solidProperties"

string_find1 = 'iterationToApplyMachineLearningPredictor'

string_insert1 = str(trainIterations)

if answer_solidsProp == "y":

    stringWrite2(filename, string_find1, string_insert1)
