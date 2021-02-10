# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Can Gözpınar 68965
# ### Engr421 Hw#5

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# %% [markdown]
# ### Divide the data set
# 

# %%
#read in data points
# read data into memory
data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",")[1:,:] #remove the first x,y string
#first 100 to training set
trainingDataSet = data_set[:100,:] #(100,2)
#remaining 33 to test set
testDataSet = data_set[100:,:] #(33,2)

# %% [markdown]
# ### Decision Tree Regression Algorithm

# %%
# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}

node_splits = {} #holds x values of the splits of the nodes
node_prediction = {} #holds predicted y value for node

# put all training instances into the root node
node_indices[1] = np.array(range(len(trainingDataSet)))
is_terminal[1] = False
need_split[1] = True


# %%
def initializeVaribles():

    #decision tree regression algorithm
    global node_indices
    global is_terminal
    global need_split
    global node_splits
    global node_prediction
    global node_indices
    global is_terminal
    global need_split
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_splits = {} #holds x values of the splits of the nodes
    node_prediction = {} #holds predicted y value for node

    # put all training instances into the root node
    node_indices[1] = np.array(range(len(trainingDataSet)))
    is_terminal[1] = False
    need_split[1] = True

# %% [markdown]
# 

# %%
def trainDecisionTree(P):
    global node_indices
    global is_terminal
    global need_split
    global node_splits
    global node_prediction
    global node_indices
    global is_terminal
    global need_split
    # learning algorithm
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:#terminate if tree is complete
            break
        
        #if tree is not complete then
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node] #elements in a given node indices
            need_split[split_node] = False #change it since we are splitting it now

            #check for current node being terminal node
            if len(data_indices) <= P: #if less than or equal to P elements in the node then terminal 
                #calculate predicted node value gm
                gm = np.mean(trainingDataSet[data_indices, 1])
                #set predicted y value for the node
                node_prediction[split_node] = gm #set prediction value for the node to gm

                is_terminal[split_node] = True#pruned

            else:#if not a terminal node then split further
                is_terminal[split_node] = False

                #decide on split positions
                unique_values = np.sort(np.unique(trainingDataSet[data_indices, 0])) #sorted all x values
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values)     - 1)]) / 2 #split positions are a-b/2 between adjacent x values namely a and b


                split_scores = np.repeat(0.0, len(split_positions))#score array for possible splits

                for s in range(len(split_positions)):#iterate over each possible split index
                        #training elements to the left of the current split
                        left_indices = data_indices[trainingDataSet[data_indices, 0] < split_positions[s]]
                        #training elements to the right of the current split
                        right_indices = data_indices[trainingDataSet[data_indices, 0] >= split_positions[s]]

                        #score calculation below: -->
                        #g function is mean for left
                        predictionLeft = np.mean(trainingDataSet[left_indices, 1])#g for left of the split
                        predictionRight = np.mean(trainingDataSet[right_indices, 1])#g for right of the     split

                        #split score is based on impurity funciton
                        split_scores[s] = 1 / len(data_indices) * ( np.sum((np.add(trainingDataSet  [left_indices, 1], predictionLeft * -1))**2) + np.sum((np.add(trainingDataSet[right_indices, 1],  predictionRight * -1))**2) ) 
                        # <-- score calculation ends here

                #choose the best split with the lowest impurity value
                best_split = split_positions[np.argmin(split_scores)]#best split x value

                #calculate predicted node value gm
                gm = np.mean(trainingDataSet[data_indices, 1])

                #set predicted y value for the node
                node_prediction[split_node] = gm #set prediction value for the node to gm
                node_splits[split_node] = best_split #set the split positon x for the node

                 # create left node using the selected split
                left_indices = data_indices[trainingDataSet[data_indices, 0] < best_split]
                node_indices[2 * split_node] = left_indices # assign lower valued for d to left
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[trainingDataSet[data_indices, 0] >= best_split]
                node_indices[2 * split_node + 1] = right_indices # assign higher valued for d to right
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

                #print( "best",best_split,"left:",left_indices,"right",right_indices)

# %% [markdown]
# 
# ### Draw test data, training data and our fit

# %%
def extractTreeInfo(P):
    #initialize variables for tree decision training first
    initializeVaribles()
    #call training tree decision funciton below to train
    trainDecisionTree(P)
    #sort the node_splits with respect to x values
    x_dict = sorted(node_splits.items(), key=lambda x : x[1])#node_index:x_position dict with sorted x_pos

    #training elements to the left of the current split
    y_values = []#array holding y values in order for x split position
    for i in range(len(x_dict) - 1):#add the middle y_prediction values
        x_Pos1 = x_dict[i][1] 
        x_Pos2 = x_dict[(i + 1)][1]
        predictionMiddle = np.mean([y for x,y in trainingDataSet if (x > x_Pos1) & (x <= x_Pos2)])
        y_values.append(predictionMiddle)

    #en sağdaki ve en soldaki eleman eksik onu da ekle son indexteki x_dict için
    #add the leftmost prediction
    y_values.insert(0, np.mean([y for x,y in trainingDataSet if x < x_dict[0][1]]))
    #add the rightmost prediction
    y_values.append(np.mean([y for x,y in trainingDataSet if x > x_dict[-1][1]]))

    x_values = [x for nodeIndex, x in x_dict] #extract x values from the list
    return x_values, y_values


# %%
#run the algorithm
P = 15
x_values, y_values = extractTreeInfo(P)

#Draw the test data, training data and our fit for P = 15 in the same figure
#plot training data on the plot
plt.figure(figsize = (10,6))

plt.plot(trainingDataSet[:,0], trainingDataSet[:,1], "b.", markersize = 10, label="training")#training
plt.plot(testDataSet[:,0], testDataSet[:,1], "r.", markersize = 10, label="test")#test set 
#plt.plot([30,40],[-50, -50], "r", markersize=10)

#our fit, prediction


#handle most left side
plt.plot([0, x_values[0]], [y_values[0], y_values[0]], "k", markersize=10)
#handle most right side    
plt.plot([x_values[-1], max(trainingDataSet[:,0])], [y_values[-1], y_values[-1]], "k", markersize=10)  

#plt.plot(x_values, y_values, "k", markersize=10)
#draw the middle y_values
for b in range(len(x_values) - 1):#draw horizontal lines
    plt.plot([x_values[b], x_values[b + 1]], [y_values[b + 1], y_values[b + 1]], "k", markersize=10)   
for b in range(len(x_values)):#draw vertical lines
    plt.plot([x_values[b], x_values[b]], [y_values[b], y_values[b + 1]], "k", markersize=10)   




plt.title("P = 15")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc = "upper left")


# %% [markdown]
# ### Calculate the root mean squared error (RMSE) 

# %%
def getRMSE(x_valuess, y_valuess):
    #Calculate the root mean squared error (RMSE) for test data points
    predictions = []#array holding predicted y values in order corresponding to test data set
    for x, y in testDataSet:
        for index in range(len(x_valuess) - 1):
            if x < x_valuess[0]:#left most y prediction
                predictions.append(y_valuess[0])
                break
            elif (x <= x_valuess[index + 1]) & (x > x_valuess[index]): #middle prediction values
                predictions.append(y_valuess[index + 1])
                break
            elif x >= x_valuess[-1]:#right most y prediction
                predictions.append(y_valuess[-1])
                break

    #apply the rmse formula
    rmse = np.sqrt(np.sum(np.add([yTest for xTest, yTest in testDataSet], -1 * np.array(predictions))**2) / len(testDataSet) )
    return rmse


# %%
#call rmse funciton to get the value
rmse = getRMSE(x_values, y_values)
print("RMSE is {} when P is 15".format(rmse))

# %% [markdown]
# ### Changing Pruning parameters and plotting it

# %%
#array holding the values of P
p_values = range(5,55,5) #array of 5,10,15...50

rmse_values = []#array to hold on rmse values for corresponding p_values value

#calculate the rmse value for each p value in the array
for p in p_values:
    x_values, y_values = extractTreeInfo(p)
    currentRMSE = getRMSE(x_values, y_values)
    rmse_values.append(currentRMSE)
    
#x ler p_values y ler de rmse values
plt.figure(figsize = (10,6))
plt.plot(p_values, rmse_values, "k", markersize=10)#draw lines
plt.plot(p_values, rmse_values, "k.", markersize=15)#draw dots    
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()


# %%



