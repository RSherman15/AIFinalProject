#CS 151 Final Project
#Emily Blatter, Michael Culhane, and Rachel Sherman

import csv
import time
import datetime
import math
from collections import Counter
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def processCustomer(rows, variableNames):
	processedRow = []

	for i in range(len(variableNames)):
		if variableNames[i] == 'time':
			times = []
			for row in rows:
				x = time.strptime(row[i], "%H:%M")
				seconds = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
				times.append(seconds)
			# averageTime = sum(times)/float(len(times))
			# processedRow.append(averageTime)
			processedRow.append(times[-1])
		
		# elif variableNames[i] == 'cost':
		# 	costs = []
		# 	for row in rows:
		# 		costs.append(int(row[i]))
		# 	averageCost = sum(costs)/float(len(costs))
		# 	processedRow.append(averageCost)
		
		# elif variableNames[i] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		# 	options = []
		# 	for row in rows:
		# 		options.append(row[i])
		# 	counter = Counter(options)
		# 	mode, _ = counter.most_common(1)[0]
		# 	processedRow.append(int(mode))

		elif variableNames[i] in ['car_value', 'state']:
			processedRow.append(rows[-1][i])

		else:
			if rows[-1][i] == 'NA':
				processedRow.append(-1)
			else: 
				processedRow.append(int(rows[-1][i]))

	return processedRow

def preprocess(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		customerID = None
		variableNames = reader.next()
		rows = []
		customers = []
		for row in reader:
			if row[0] != customerID:
				if len(rows) > 0:
					customers.append(processCustomer(rows, variableNames))
				customerID = row[0]
				rows = []
			rows.append(row)
		if len(rows) > 0:
			customers.append(processCustomer(rows, variableNames))			

	customers = np.array(customers)
	return (customers, variableNames)

def main():
	trainingCustomers, variableNames = preprocess('train.csv')
	testCustomers, _ = preprocess('test_v2.csv')

	stateEncoder = preprocessing.LabelEncoder()
	carValueEncoder = preprocessing.LabelEncoder()
	trainStates = trainingCustomers[:,variableNames.index('state')]
	testStates = testCustomers[:,variableNames.index('state')]
	stateStack = np.hstack((trainStates, testStates))
	stateEncoder.fit(stateStack)

	trainCarValues = trainingCustomers[:,variableNames.index('car_value')]
	testCarValues = testCustomers[:,variableNames.index('car_value')]
	carValueStack = np.hstack((trainCarValues, testCarValues))
	carValueEncoder.fit(carValueStack)

	encodedTrainStates = stateEncoder.transform(trainStates)
	encodedTrainCarValues = carValueEncoder.transform(trainCarValues)
	encodedTestStates = stateEncoder.transform(testStates)
	encodedTestCarValues = carValueEncoder.transform(testCarValues)


	trainingCustomers[:,variableNames.index('state')] = encodedTrainStates
	trainingCustomers[:,variableNames.index('car_value')] = encodedTrainCarValues
	testCustomers[:,variableNames.index('state')] = encodedTestStates
	testCustomers[:,variableNames.index('car_value')] = encodedTestCarValues


	outputs = []
	for outputCol in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		trainOutput = trainingCustomers[:,variableNames.index(outputCol)]
		trainInput = np.delete(trainingCustomers, variableNames.index(outputCol), 1)

		testOutput = testCustomers[:,variableNames.index(outputCol)]
		testInput = np.delete(testCustomers, variableNames.index(outputCol), 1)

		classifier = KNeighborsClassifier(n_neighbors = 30)
		#classifier = SVC()
		classifier.fit(trainInput[:,1:], trainOutput)
		outputs.append(classifier.predict(testInput[:,1:]))
		print 'output column: ', outputCol
		print 'score: ', classifier.score(testInput[:,1:], testOutput)

	resultStrings = []
	customerIDs = testCustomers[:, variableNames.index('customer_ID')]
	for customerID, customerResult in zip(customerIDs, zip(*outputs)):
		resultString = customerID + ',' + ''.join([x for x in customerResult])
		resultStrings.append(resultString)

	with open('results.csv', 'w') as f:
		f.write('customer_ID,plan\n')
		f.write('\n'.join(resultStrings))

if __name__ == "__main__":
	main()



