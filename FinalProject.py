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
from sklearn import metrics
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
PRINT_METRICS = True

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

			if len(rows) >= 2:
				processedRow.append(rows[lastViewedIndex-1][i])
			else:
				processedRow.append(0)

			if len(rows) >= 3:
				processedRow.append(rows[lastViewedIndex-2][i])
			else:
				processedRow.append(0)

			# options = []
			# for row in rows:
			# 	options.append(row[i])
			# counter = Counter(options)
			# mode, _ = counter.most_common(1)[0]
			# processedRow.append(int(mode))


		elif variableNames[i] in ['car_value', 'state']:
			processedRow.append(rows[lastViewedIndex][i])

		else:
			if rows[lastViewedIndex][i] == 'NA':
				processedRow.append(0)
			else: 
				processedRow.append(rows[lastViewedIndex][i])

	return processedRow

def processPurchasePoint(lastViewedRow, actuallyPurchasedRow, variableNames):
	boughtLastViewed = True
	purchasedOptions = []
	for variable in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		if lastViewedRow[variableNames.index(variable)] != actuallyPurchasedRow[variableNames.index(variable)]:
			boughtLastViewed = False
		purchasedOptions.append(actuallyPurchasedRow[variableNames.index(variable)])
	return boughtLastViewed, purchasedOptions


def preprocess(filename, isTrain):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		customerID = None
		variableNames = reader.next()

		rows = []
		customers = []
		purchasedLastViewed = []
		purchasedOptions = []
		for row in reader:
			if row[0] != customerID:
				if len(rows) > 0:
					if isTrain:
						customers.append(processCustomer(rows[:-1], variableNames))
						didPurchaseLastViewed, optionsPurchased = processPurchasePoint(rows[-2], rows[-1], variableNames)
						purchasedLastViewed.append(didPurchaseLastViewed)
						purchasedOptions.append(optionsPurchased)						
					else:
						customers.append(processCustomer(rows, variableNames))
				customerID = row[0]
				rows = []
			rows.append(row)
		if len(rows) > 0:
			if isTrain:
				customers.append(processCustomer(rows[:-1], variableNames))
				didPurchaseLastViewed, optionsPurchased = processPurchasePoint(rows[-2], rows[-1], variableNames)
				purchasedLastViewed.append(didPurchaseLastViewed)
				purchasedOptions.append(optionsPurchased)
			else:
				customers.append(processCustomer(rows, variableNames))			

	for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		variableNames.insert(variableNames.index(x)+1, x+'_old2')
		variableNames.insert(variableNames.index(x)+1, x+'_old1')

	customers = np.array(customers)
	purchasedLastViewed = np.array(purchasedLastViewed)
	purchasedOptions = np.array(purchasedOptions)
	return customers, variableNames, purchasedLastViewed, purchasedOptions

def lastViewedClassifier(trainInput, trainOutput, testInput, variableNames):
	predictedTestOutputs = []
	predictedTrainOutputs = []

	#classifier = KNeighborsClassifier(n_neighbors = 30)
	classifier = GaussianNB()
	classifier.fit(trainInput[:,1:], trainOutput)
	predictedTrainOutputs.extend(classifier.predict(trainInput[:,1:]))
	predictedTestOutputs.extend(classifier.predict(testInput[:,1:]))

	if PRINT_METRICS:
		# Randomly split the data into training and testing sets for validation.
		train_in, test_in, train_out, test_out = cross_validation.train_test_split(
			trainInput[:,1:], trainOutput, test_size=0.20)

		# Train a classifier on the train data and validate the results with the
		# testing set.
		# NOTE: The stats printed here aren't going to be 100% accurate, as we're
		# actually incorporating all of the data in our actual classification.
		#stat_classifier = KNeighborsClassifier(n_neighbors = 30)
		stat_classifier = GaussianNB()
		stat_classifier.fit(train_in, train_out)
		predictions = stat_classifier.predict(test_in)
		print "\n\nPurchased Last Viewed Classifier"
		print metrics.classification_report(test_out, predictions)
		print "Accuracy", metrics.accuracy_score(test_out, predictions)		

	resultString = ""
	testIndicesToDelete = []
	for i in range(len(predictedTestOutputs)):
		if predictedTestOutputs[i] == True:
			resultString += str(int(testInput[i][0])) + ","
			for coverageOption in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
				resultString += str(int(testInput[i][variableNames.index(coverageOption)]))
			resultString += '\n'
			testIndicesToDelete.append(i)

	trainIndicesToDelete = []
	for i in range(len(predictedTrainOutputs)):
		if predictedTrainOutputs[i] == True:
			trainIndicesToDelete.append(i)

	return resultString, trainIndicesToDelete, testIndicesToDelete

def secondClassifier(trainInput, purchasedOptions, testInput, variableNames):
	'''this classifier is significantly worse than the last. Probably because we don't really know yet???'''
	outputs = []
	for i in range(len(['A', 'B', 'C', 'D', 'E', 'F', 'G'])):
		trainOutput = purchasedOptions[:,i]

		# classifier = KNeighborsClassifier(n_neighbors = 30)
		# classifier = SVC()
		classifier = GaussianNB()
		classifier.fit(trainInput[:,1:], trainOutput)
		testPredictions = classifier.predict(testInput[:,1:])
		outputs.append(testPredictions)

		trainPredictions = classifier.predict(trainInput[:,1:])

		testInput[:,variableNames.index(['A', 'B', 'C', 'D', 'E', 'F', 'G'][i])] = testPredictions
		trainInput[:,variableNames.index(['A', 'B', 'C', 'D', 'E', 'F', 'G'][i])] = trainPredictions



		if PRINT_METRICS:
			# Randomly split the data into training and testing sets for validation.
			train_in, test_in, train_out, test_out = cross_validation.train_test_split(
				trainInput[:,1:], trainOutput, test_size=0.20)

			# Train a classifier on the train data and validate the results with the
			# testing set.
			# NOTE: The stats printed here aren't going to be 100% accurate, as we're
			# actually incorporating all of the data in our actual classification.
			#stat_classifier = KNeighborsClassifier(n_neighbors = 30)
			stat_classifier = GaussianNB()
			stat_classifier.fit(train_in, train_out)
			predictions = stat_classifier.predict(test_in)
			print "\n\nPlan", ['A', 'B', 'C', 'D', 'E', 'F', 'G'][i], "Classifier"
			print metrics.classification_report(test_out, predictions)
			print "Accuracy", metrics.accuracy_score(test_out, predictions)

	resultStrings = []
	customerIDs = testInput[:, variableNames.index('customer_ID')]
	for customerID, customerResult in zip(customerIDs, zip(*outputs)):
		resultString = str(int(customerID)) + ',' + ''.join([str(int(x)) for x in customerResult])
		resultStrings.append(resultString)

	return '\n'.join(resultStrings)

def main():
	trainingCustomers, variableNames, purchasedLastViewed, purchasedOptions = preprocess('train.csv', True)
	testCustomers, _, _, _ = preprocess('test_v2.csv', False)

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

	trainInput = trainInput.astype(float)
	testInput = testInput.astype(float)

	# resultString, trainIndicesToDelete, testIndicesToDelete = lastViewedClassifier(trainInput, trainOutput, testInput, variableNames)

	# trainInput = np.delete(trainInput, trainIndicesToDelete, axis=0)
	# purchasedOptions = np.delete(purchasedOptions, trainIndicesToDelete, axis=0)
	# testInput = np.delete(testInput, testIndicesToDelete, axis=0)

	resultString = ""
	resultString += secondClassifier(trainInput, purchasedOptions, testInput, variableNames)



	with open('results.csv', 'w') as f:
		f.write('customer_ID,plan\n')
		f.write(resultString)

if __name__ == "__main__":
	main()



