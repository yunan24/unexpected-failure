import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import csv
import random

from csv import writer
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error





def predict_random(Xtest, model): #predicts and prints a random datapoint from testing dataset
	num = 485
	sample = Xtest.sample(n=1, random_state=np.random.randint(low=0, high=num))
	idx = sample.index[0]
	BP=[]
	i = 0
	while i <= 24:
		BPi = dataset.iloc[idx, i]
		BP.append(BPi)
		i = i+1
	TargetRankings=[]
	i=25
	while i <= 33:
		TRankingsi = dataset.iloc[idx, i]
		TargetRankings.append(TRankingsi)
		i = i+1
	idx = idx + 2
	ninput = (BP)
	ninput_array = np.array(ninput)
	PredictedRankings = np.round(model.predict(ninput_array.reshape(1,-1)),2)
	print("PREDICT RANDOM------------------------------------------------------------------------------")
	print("\nIndex: "+str(idx))
	print("Board Position: " + str(BP))
	print("Target Rankings:    " + str(TargetRankings))
	print("Predicted Rankings: " + str(PredictedRankings))
	
def testtrials(model):
	print("\n\n\n\nTESTING TRIALS-----------------------------------------------------------------------------------")
	trial_file = open("test_trials.csv", "r")
	triallines = trial_file.readlines()
	NUM_TestTRIALS = 0
	for x in triallines:
		NUM_TestTRIALS = NUM_TestTRIALS + 1

	index = 0
	trialBP = []
	while index <= NUM_TestTRIALS-1:
		trialBP = []
		x = triallines[index].rstrip('\n')
		trialline = x.split(',')
		i=0
		while i<=24:
			trialBP.append(int(float(trialline[i])))
			i = i+1
		
		ninput = (trialBP)
		ninput_array = np.array(ninput)
		print(ninput)
		PredictedRankings = np.round(model.predict(ninput_array.reshape(1,-1)),2)
		print("\nBoard Position: " + str(trialBP))
		print("Predicted Rankings: " + str(PredictedRankings))
		print("\n------------------------------\n")
		
		index = index + 1



def testdatasets(model, Xtest, ytest):	
	resultlist = list()
	evaluatemodel = model.evaluate(Xtest, ytest, verbose=1)
	resultlist.append(evaluatemodel)
	results = resultlist
	print('\nTesting MSE: %.3f' % (mean(results)))
	mse = '%.3f' % (mean(results))
	return float(mse)
#------------------------------------------------------------------------------------------------------------------------
#DATASET STUFF

DATASET = 'dataset2_5by5_training.csv'
TESTING_DATASET = 'dataset2_5by5_testing.csv'
FINAL_TESTING_DATASET = 'final_testing_dataset.csv'
FINAL_NUM_ROWS = 607 #number of rows in final dataset
NUM_ROWS_IN_DATASET = 2423

#------------------------------------------------------------------------------------------------------------------------
# SIMILARITY TRAINING (CALCULATES MAX AND MIN VALUES FOR EACH INPUT)
	
class InputPosition:
	def __init__(self, index, max_value, min_value):
		self.index = index
		self.max = max_value
		self.min = min_value
		
column_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','','','','','','','','','','']
input_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
dataset = pd.read_csv(DATASET, nrows=NUM_ROWS_IN_DATASET, names=column_names)

InputClassList = []

for i in input_names:
	input_list = dataset[i].values.tolist()
	index = ord(i) - 97
	max_value = max(input_list)
	min_value = min(input_list)
	
	
	
	if i == 'a':
		index_0 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_0)
	elif i == 'b':
		index_1 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_1)
	elif i == 'c':
		index_2 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_2)
	elif i == 'd':
		index_3 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_3)
	elif i == 'e':
		index_4 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_4)
	elif i == 'f':
		index_5 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_5)
	elif i == 'g':
		index_6 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_6)
	elif i == 'h':
		index_7 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_7)
	elif i == 'i':
		index_8 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_8)
	elif i == 'j':
		index_9 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_9)
	elif i == 'k':
		index_10 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_10)
	elif i == 'l':
		index_11 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_11)
	elif i == 'm':
		index_12 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_12)
	elif i == 'n':
		index_13 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_13)
	elif i == 'o':
		index_14 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_14)
	elif i == 'p':
		index_15 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_15)
	elif i == 'q':
		index_16 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_16)
	elif i == 'r':
		index_17 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_17)
	elif i == 's':
		index_18 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_18)
	elif i == 't':
		index_19 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_19)
	elif i == 'u':
		index_20 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_20)
	elif i == 'v':
		index_21 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_21)
	elif i == 'w':
		index_22 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_22)
	elif i == 'x':
		index_23 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_23)
	elif i == 'y':
		index_24 = InputPosition(index, max_value, min_value)
		InputClassList.append(index_24)



### Load Normal Dataset
dataset = pd.read_csv(DATASET, nrows=NUM_ROWS_IN_DATASET)
train, test = train_test_split(dataset, test_size = 0.2, random_state=42)

X_train = train.iloc[:, 0:25]
y_train = train.iloc[:, 25:34]

a = test.to_csv(index=False)
afile = open('unmodified_normal_testing_sample.csv', 'w+')
afile.write(a)
afile.close()
with open('unmodified_normal_testing_sample.csv', 'r') as original_a:
	with open('normal_testing_sample.csv', 'w+') as new_a:
		next(original_a)
		for line in original_a:
			new_a.write(line)

### Load Failure dataset
datasetcf = pd.read_csv(TESTING_DATASET, nrows=NUM_ROWS_IN_DATASET)
traincf, testcf = train_test_split(datasetcf, test_size = 0.05, random_state=42)
b = testcf.to_csv(index=False)
bfile = open('unmodified_failure_testing_sample.csv', 'w+')
bfile.write(b)
with open('unmodified_failure_testing_sample.csv', 'r') as original_b:
	with open('failure_testing_sample.csv', 'w+') as new_b:
		next(original_b)
		for line in original_b:
			new_b.write(line)

### Create Final Dataset
a_reader = csv.reader(open('normal_testing_sample.csv'))
b_reader = csv.reader(open('failure_testing_sample.csv'))
final_dataset = open('final_testing_dataset.csv', 'w+')
writer = csv.writer(final_dataset)

for row in a_reader:
	writer.writerow(row)
for row in b_reader:
	writer.writerow(row)
	
final_dataset.close()

#Remove first line in normal and failure dataset


##--------------------------------------------------------------------------------------------------------------
#MODEL ARCHITECTURES

print('\n\nMODEL 1 TRAINING-----------------------------------------------------------------------------------')
#model 1 architecture
model = Sequential()
model.add(Dense(25, input_dim=25, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation= 'sigmoid'))
model.compile(loss='mse', optimizer='adam',)
model.fit(X_train, y_train, verbose=2, epochs=200)
							  
							  

#------------------------------------------------------------------------------------------------------------------------

#SIMILARITY TESTING (SPLITS DATA)



def append_list_as_row(filename, list_of_elem):
	with open(filename, 'a+', newline='') as write_obj:
		csv_writer = csv.writer(write_obj)
		csv_writer.writerow(list_of_elem)


def testmodel3(testing_dataset):
	dataset_model_1 = open('Model5_Splitdata1.csv', 'w+') #no outliers
	dataset_model_2 = open('Model5_Splitdata2.csv', 'w+') #outliers
	#split data by similarity
	for line in open(testing_dataset):
		csv_row_full = line.split(',')
		csv_row_board = csv_row_full[:25]
		csv_row_board = list(map(int, csv_row_board))
		outlierboolean = False
		#if you detect outlier, compress
		j=0
		outlier = 0 #outlier_num (0 if not an outlier)
		for i in csv_row_board:
			max_val = InputClassList[j].max
			min_val = InputClassList[j].min	
			if i > max_val+3:
				outlierboolean = True
				outlier = i	
			elif i < min_val:
				outlierboolean = True
				outlier = i
			j = j+1
			if outlierboolean ==True:
				x = outlier - 1
				y = np.log(x) + 2
				compressed_input = round(y, 2)
				csv_row_full = [str(compressed_input) if a==str(outlier) else a for a in csv_row_full]
				csv_row_full.pop()
				break
				
		if outlierboolean == True: #True=outliers False = no outliers
			append_list_as_row('Model5_Splitdata2.csv', csv_row_full)
		elif outlierboolean == False:
			append_list_as_row('Model5_Splitdata1.csv', csv_row_full)


	#feed data to models
	MODEL_1_DATASET = 'Model5_Splitdata1.csv'
	MODEL_2_DATASET = 'Model5_Splitdata2.csv'
	row_count_1 = len(list(csv.reader(open(MODEL_1_DATASET))))
	row_count_2 = len(list(csv.reader(open(MODEL_2_DATASET))))



	
	model_1_data_empty = os.stat('Model5_Splitdata1.csv').st_size == 0 #True if empty, False if not
	model_2_data_empty = os.stat('Model5_Splitdata2.csv').st_size == 0
	
	if model_1_data_empty == False:
		dataset_model_1 = pd.read_csv(MODEL_1_DATASET, nrows=row_count_1)
		X_test_1 = dataset_model_1.iloc[:, 0:25]
		y_test_1 = dataset_model_1.iloc[:, 25:34]
		Model1MSE = testdatasets(model, X_test_1, y_test_1)
	else:
		Model1MSE = 0
		print('\n\nDataset 1 - No data')
		
	if model_2_data_empty == False:
		dataset_model_2 = pd.read_csv(MODEL_2_DATASET, nrows=row_count_2)
		X_test_2 = dataset_model_2.iloc[:, 0:25]
		y_test_2 = dataset_model_2.iloc[:, 25:34]
		Model2MSE = testdatasets(model, X_test_2, y_test_2)
	else:
		Model2MSE = 0
		print('\n\nDataset 2 - No data')

	#combine results

	weightedm1 = Model1MSE * row_count_1
	weightedm2 = Model2MSE * row_count_2
	combinedweightedMSE = weightedm1 +weightedm2
	total_row_count = row_count_1 + row_count_2 - 2
	FinalMSE = combinedweightedMSE / total_row_count
	print("\n\n-------------------------------------------------------------------------------------------------------")
	print('Final MSE for Model 5: %.3f' %(FinalMSE))
			
			
			
print("-------------------------------------------------------------------------------------------------------")
print("FINAL TESTING DATA")
testmodel3(FINAL_TESTING_DATASET)

print("-------------------------------------------------------------------------------------------------------")
print("NORMAL DATA")
testmodel3('normal_testing_sample.csv')
print("-------------------------------------------------------------------------------------------------------")
print("FAILURE DATA")
testmodel3('failure_testing_sample.csv')
print("-------------------------------------------------------------------------------------------------------")


