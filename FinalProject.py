#CS 151 Final Project
#Emily Blatter, Michael Culhane, and Rachel Sherman

import csv
import time
import datetime
import math
from collections import Counter
import numpy as np
import scipy
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def processCustomer(rows, variableNames):
	processedRow = []
	lastViewed = ""
	actuallyPurchased = ""
	lastViewedIndex = -1

	for i in range(len(variableNames)):
		if variableNames[i] == 'time':
			times = []
			for row in rows:
				x = time.strptime(row[i], "%H:%M")
				seconds = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
				times.append(seconds)
			processedRow.append(times[lastViewedIndex])
				
		elif variableNames[i] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
			processedRow.append(rows[lastViewedIndex][i])


		elif variableNames[i] in ['car_value', 'state']:
			processedRow.append(rows[lastViewedIndex][i])

		else:
			if rows[lastViewedIndex][i] == 'NA':
				processedRow.append(-1)
			else: 
				processedRow.append(int(rows[lastViewedIndex][i]))

	return processedRow

def processPurchasePoint(lastViewedRow, actuallyPurchasedRow, variableNames):
	for variable in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		if lastViewedRow[variableNames.index(variable)] != actuallyPurchasedRow[variableNames.index(variable)]:
			return False
	return True


def preprocess(filename, isTrain):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		customerID = None
		variableNames = reader.next()
		rows = []
		customers = []
		purchasedLastViewed = []
		for row in reader:
			if row[0] != customerID:
				if len(rows) > 0:
					if isTrain:
						customers.append(processCustomer(rows[:-1], variableNames))
						purchasedLastViewed.append(processPurchasePoint(rows[-2], rows[-1], variableNames))
					else:
						customers.append(processCustomer(rows, variableNames))
				customerID = row[0]
				rows = []
			rows.append(row)
		if len(rows) > 0:
			print rows
			if isTrain:
				customers.append(processCustomer(rows[:-1], variableNames))
				purchasedLastViewed.append(processPurchasePoint(rows[-2], rows[-1]))
			else:
				customers.append(processCustomer(rows, variableNames))			

	customers = np.array(customers)
	return (customers, variableNames, purchasedLastViewed)

def lastViewedClassifier(trainInput, trainOutput, testInput, variableNames):
	predictedTestOutputs = []
	predictedTrainOutputs = []

	classifier = KNeighborsClassifier(n_neighbors = 30)
	classifier.fit(trainInput[:,1:], trainOutput)
	predictedTrainOutputs.extend(classifier.predict(trainInput[:,1:]))
	predictedTestOutputs.extend(classifier.predict(testInput[:,1:]))

	resultString = ""
	testIndicesToDelete = []
	for i in range(len(predictedTestOutputs)):
		if predictedTestOutputs[i] == 'True':
			resultString += str(testInput[i][0]) + ","
			for coverageOption in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
				resultString += str(testInput[i][variableNames.index(coverageOption)])
			resultString += '\n'
			testIndicesToDelete.append(i)

	trainIndicesToDelete = []
	for i in range(len(predictedTrainOutputs)):
		if predictedTrainOutputs[i] == 'True':
			trainIndicesToDelete.append(i)

	return resultString, trainIndicesToDelete, testIndicesToDelete

def secondClassifier(trainData, testInput, variableNames):
	'''this one is significantly shittier than the last. Probably because we don't really know yet???'''
	outputs = []
	for outputCol in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		trainOutput = trainData[:,variableNames.index(outputCol)]
		trainInput = np.delete(trainData, variableNames.index(outputCol), 1)

		testOutput = testCustomers[:,variableNames.index(outputCol)]
		testInput = np.delete(testCustomers, variableNames.index(outputCol), 1)

		classifier = KNeighborsClassifier(n_neighbors = 30)
		# classifier = SVC()
		classifier.fit(trainInput[:,1:], trainOutput)
		outputs.append(classifier.predict(testInput[:,1:]))
		print 'output column: ', outputCol

	resultStrings = []
	customerIDs = testCustomers[:, variableNames.index('customer_ID')]
	for customerID, customerResult in zip(customerIDs, zip(*outputs)):
		resultString = customerID + ',' + ''.join([x for x in customerResult])
		resultStrings.append(resultString)

	return '\n'.join(resultStrings)

def main():
	trainingCustomers, variableNames, purchasedLastViewed = preprocess('train.csv', True)
	testCustomers, _, _ = preprocess('test_v2.csv', False)

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

	trainOutput = purchasedLastViewed
	trainInput = trainingCustomers
	testInput = testCustomers

	resultString, trainIndicesToDelete, testIndicesToDelete = lastViewedClassifier(trainInput, trainOutput, testInput, variableNames)

	trainData = numpy.delete(trainInput, trainIndicesToDelete)
	testInput = numpy.delete(testInput, testIndicesToDelete)

	resultString += secondClassifier(trainData, testInput, variableNames)



	with open('results.csv', 'w') as f:
		f.write('customer_ID,plan\n')
		f.write(resultString)

if __name__ == "__main__":
	main()



