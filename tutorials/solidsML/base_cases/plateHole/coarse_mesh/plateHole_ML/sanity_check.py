import numpy as np

prediction_ML = np.array(np.loadtxt("../../machine_learning/prediction.csv", delimiter=","))

def xyz(filepath):

    my_file = open(filepath)
    string_list = my_file.readlines()
    index=22
    size=1000
    string_list = string_list[index:len(string_list)] 
    string_list = string_list[0:size] 
    i=0
    for i in range(len(string_list)):
        string_list[i]=string_list[i].replace("(","")
        string_list[i]=string_list[i].replace(")","")
        string_list[i]=string_list[i].split()
    arr=np.array(string_list)
    arr= arr.astype(np.float64)

    return arr

prediction_OF = xyz("1/D_predicted")

# shape = (1000, 3)

difference = prediction_ML - prediction_OF

import sigfig


for i in np.arange(8,0,-1):

    diff_x = 0
   
    for j in np.arange(0, len(prediction_ML[:,0])):

        prediction_ML_rounded = sigfig.round(prediction_ML[j][0], sigfigs=i)
        prediction_OF_rounded = sigfig.round(prediction_OF[j][0], sigfigs=i) 

        difference_temp = prediction_ML_rounded - prediction_OF_rounded

        if ((np.sqrt(difference_temp**2)) > 0):        
            diff_x  = diff_x + 1
        
    print(str(1- diff_x/len(prediction_ML[:,0])) + " percent of x were correct to " + str(i) + " figures")












# difference_x = difference[:, 0]
# difference_y = difference[:, 1]
# difference_z = difference[:, 2]

# prediction_ML_x = prediction_ML[:, 0]
# prediction_ML_y = prediction_ML[:, 1]
# prediction_ML_z = prediction_ML[:, 2]

# prediction_OF_x = prediction_OF[:, 0]
# prediction_OF_y = prediction_OF[:, 1]
# prediction_OF_z = prediction_OF[:, 2]

# bad = 0

# for i in np.arange(0, len(difference_x)):

#     factor = prediction_ML_x[i]%10   

#     # Counts how many figures are the same

#     n=int(1/difference_x[i])
#     count=0
#     while(n>0):
#         count=count+1
#         n=n//10
#     count1 = count

#     n=int(1/prediction_ML_x[i])
#     count=0
#     while(n>0):
#         count=count+1
#         n=n//10
#     count2 = count

#     diff = np.sqrt((count1-count2)**2)

#     if diff < 6 :
#         bad = bad +1
    
#         print(i)
#         print(diff)
#         print(difference_x[i])
#         print(prediction_ML_x[i])
#         print(prediction_OF_x[i])

# print("total " + str(bad))



