import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import csv
import random

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
		
		
		
		
		
		
		
		




DATASET = 'dataset2_5by5_training.csv'
TESTING_DATASET = 'dataset2_5by5_testing.csv'
FINAL_TESTING_DATASET = 'final_testing_dataset.csv'
FINAL_NUM_ROWS = 607
NUM_ROWS_IN_DATASET = 2423


### Load Normal Dataset
dataset = pd.read_csv(DATASET, nrows=NUM_ROWS_IN_DATASET)
train, test = train_test_split(dataset, test_size = 0.2, random_state=42)


X_train = train.iloc[:, 0:25]
y_train = train.iloc[:, 25:34]
X_test = test.iloc[:, 0:25]
y_test = test.iloc[:, 25:34]

a = test.to_csv(index=False)

afile = open('normal_testing_sample.csv', 'w+')
afile.write(a)

### Load Failure dataset
datasetcf = pd.read_csv(TESTING_DATASET, nrows=NUM_ROWS_IN_DATASET)
traincf, testcf = train_test_split(datasetcf, test_size = 0.05, random_state=42)


X_traincf = traincf.iloc[:, 0:25]
y_traincf = traincf.iloc[:, 25:34]
X_testcf = testcf.iloc[:, 0:25]
y_testcf = testcf.iloc[:, 25:34]

b = testcf.to_csv(index=False)
bfile = open('failure_testing_sample.csv', 'w+')
bfile.write(b)

### Create Final Dataset
a_reader = csv.reader(open('normal_testing_sample.csv'))
next(a_reader)
b_reader = csv.reader(open('failure_testing_sample.csv'))
next(b_reader)
final_dataset = open('final_testing_dataset.csv', 'w+')
writer = csv.writer(final_dataset)

for row in a_reader:
	writer.writerow(row)
for row in b_reader:
	writer.writerow(row)
	
final_dataset.close()

### Load Final Dataset
datasetF = pd.read_csv(FINAL_TESTING_DATASET, nrows=FINAL_NUM_ROWS)
X_testF = datasetF.iloc[:, 0:25]
y_testF = datasetF.iloc[:, 25:34]
	
	
#model architecture
model = Sequential()
model.add(Dense(256, input_dim=25, kernel_initializer='normal', activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation= 'sigmoid'))
model.compile(loss='mse', optimizer='adam',)
model.fit(X_train, y_train, verbose=2, epochs=200)


print("\n\nNORMAL TESTING")
testdatasets(model, X_test, y_test)

print("\n\nCATASTROPHIC TESTING")
testdatasets(model, X_testcf, y_testcf)

print("\n\nFINAL TESTING")
testdatasets(model, X_testF, y_testF)
	
#predict_random(X_test, model)
testtrials(model)

