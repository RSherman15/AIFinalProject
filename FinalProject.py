# CS 151 Final Project
# Emily Blatter, Michael Culhane, and Rachel Sherman
# Friday, May 9, 2014

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

# Flag that controls whether metrics (precision, recall, accuracy)
# should be printed for each classifier.
PRINT_METRICS = True

# Specifies the name of the output CSV. The output will be formatted
# in a way that the output can be directly submitted to Kaggle.
OUTPUT_FILE_NAME = 'results.csv'


def processCustomer(rows, variableNames):
	'''
	Does the preprocessing for one customer. Takes in an array containing all
	of the rows of the initial data for a given customer, as well as an array
	variableNames indicating the index of each field in the data.

	Collapses the rows into one row of features for the customer.
	'''

	processedRow = []

	# Index in rows of the last row viewed by the customer.
	lastViewedIndex = -1

	# Collapse each column one at a time.
	for i in range(len(variableNames)):

		# For the time column, we convert each time to a datetime and take
		# the delta from the epoch (in seconds).
		# In our collapsed row, we just take the time of the last shopping
		# trip (though, in the future, we could do more intricate averaging
		# if desired).
		if variableNames[i] == 'time':
			times = []
			for row in rows:
				x = time.strptime(row[i], "%H:%M")
				seconds = datetime.timedelta(hours=x.tm_hour, 
											 minutes=x.tm_min, 
											 seconds=x.tm_sec).total_seconds()
				times.append(seconds)
			processedRow.append(times[lastViewedIndex])
		
		# For each policy we append to our collapsed row the values from
		# the last three shopping trips. If there are only 1 or 2 shopping
		# trips, we insert 0 as a dummy value.
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

		# For our two string columns, we simply append the value to the
		# feature vector.
		elif variableNames[i] in ['car_value', 'state']:
			processedRow.append(rows[lastViewedIndex][i])

		# For all other columns, we simply append the value from the last
		# shopping point. In the case where we encounter 'NA', we append
		# 0 as a dummy value.
		# 
		# Even though -1 may make more sense as a dummy value (especially
		# considering that some fields may actually have a possible value
		# of 0), the classifiers can't seem to handle negative numbers in
		# the training data. Thus, we use 0.
		else:
			if rows[lastViewedIndex][i] == 'NA':
				processedRow.append(0)
			else: 
				processedRow.append(rows[lastViewedIndex][i])

	return processedRow

def processPurchasePoint(lastViewedRow, actuallyPurchasedRow, variableNames):
	'''
	Processes one purchase point (which is just one row in the training data)
	to obtain the corresponding outputs for each of the classifiers.

	The first value returned is boughtLastViewed, which specifies whether the
	customer bought the last plan they shopped. (More explicitly, we Check
	whether the lastViewedRow matches the actuallyPurchasedRow for each of
	A-G).

	The second value returned is an array containing the value of each option
	that the customer purchased. This data comes exclusively from the
	actuallyPurchasedRow.

	As in processCustomer, variableNames is simply an array of the names of
	the columns in the data, ordered exactly as the data are ordered in
	lastViewedRow and actuallyPurchasedRow.
	'''
	boughtLastViewed = True
	purchasedOptions = []
	for variable in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		# Check the value in the lastViewedRow against the value in the
		# actuallyPurchasedRow. If different, the user did not buy the
		# last thing they shopped for.
		if (lastViewedRow[variableNames.index(variable)] != 
				actuallyPurchasedRow[variableNames.index(variable)]):
			boughtLastViewed = False
		purchasedOptions.append(
			actuallyPurchasedRow[variableNames.index(variable)])
	return boughtLastViewed, purchasedOptions


def preprocess(filename, isTrain):
	'''
	Preprocesses a file by reading in the data, preprocessing the rows for
	each customer, and returning an array of processed data.
	'''

	with open(filename, 'r') as f:

		# The CSV reader parses the file row by row, separating
		# values at the provided delimeter.
		reader = csv.reader(f, delimiter=',')
		customerID = None

		# Grab the first row of the file, which contains the variable names.
		variableNames = reader.next()

		rows = []
		customers = []
		purchasedLastViewed = []
		purchasedOptions = []

		# Iterate through each line in the file.
		for row in reader:
			# If we're at a new customer, preprocess the rows we've collected
			# so far.
			if row[0] != customerID:
				if len(rows) > 0:
					if isTrain:
						# Preprocess all of the shopping points (all rows
						# except the last).
						customers.append(
							processCustomer(rows[:-1], variableNames))

						# Preprocess the purchase point and store the results.
						didPurchaseLastViewed, optionsPurchased = 
							processPurchasePoint(
								rows[-2], rows[-1], variableNames)
						purchasedLastViewed.append(didPurchaseLastViewed)
						purchasedOptions.append(optionsPurchased)						
					else:
						# If this is testing data, we process all rows
						# (because the file contains no purchase point
						# that needs to be processed separately).
						customers.append(processCustomer(rows, variableNames))

				# Figure out who the next customer is.
				customerID = row[0]
				rows = []
			rows.append(row)

		# Once we've seen all the rows, make sure to preprocess any rows for
		# the last customer that we haven't yet preprocessed.
		if len(rows) > 0:
			if isTrain:
				customers.append(processCustomer(rows[:-1], variableNames))
				didPurchaseLastViewed, optionsPurchased = 
					processPurchasePoint(rows[-2], rows[-1], variableNames)
				purchasedLastViewed.append(didPurchaseLastViewed)
				purchasedOptions.append(optionsPurchased)
			else:
				customers.append(processCustomer(rows, variableNames))			

	# In processCustomers, "_old1" and "_old2" columns were added to
	# hold the values from the second- and third-to-last shopping points.
	# Here, we update variableNames to account for those changes.
	for x in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		variableNames.insert(variableNames.index(x)+1, x+'_old2')
		variableNames.insert(variableNames.index(x)+1, x+'_old1')

	# The classifiers rely on NumPy arrays, so we convert our arrays of
	# inputs/outputs into the NumPy equivalents.
	customers = np.array(customers)
	purchasedLastViewed = np.array(purchasedLastViewed)
	purchasedOptions = np.array(purchasedOptions)

	return customers, variableNames, purchasedLastViewed, purchasedOptions

def lastViewedClassifier(trainInput, trainOutput, testInput, variableNames):
	'''
	Classifier to determine which customers purchased the plan they last
	shopped.

	Takes in the processed input and output for the training set and the
	processed input for the testing set (whose output is unknown).

	Returns three values:
		resultString - CSV-format string containing the properly-formatted
		               result for each customer who was predicted to have
		               purchased their last shopped plan.
		trainIndicesToDelete - Array of indices in the training input and
		                       output corresponding with customers who
		                       were predicted to purchase their last-shopped
		                       plan. These arrays should be deleted from the
		                       training data before continuing with future
		                       classifications.
   		testIndicesToDelete - Array of indices in the training input and
		                      output corresponding with customers who
		                      were predicted to purchase their last-shopped
		                      plan. These arrays should be deleted from the
		                      training data before continuing with future
		                      classifications.
	'''

	predictedTestOutputs = []
	predictedTrainOutputs = []

	classifier = KNeighborsClassifier(n_neighbors = 30)
	# classifier = GuassianNB()

	# Fit the classifier with the trainInput and trainOutput. We leave off
	# the first column of the trainInput, which just contains the customerID
	# (a field that probably isn't too useful for classification).
	classifier.fit(trainInput[:,1:], trainOutput)

	# Using the trained classifier, figure out which values are predicted
	# for both the training input and the testing input.
	predictedTrainOutputs.extend(classifier.predict(trainInput[:,1:]))
	predictedTestOutputs.extend(classifier.predict(testInput[:,1:]))

	# If requested, print some metrics about the result of the classification.
	# We do this using a new classifier which has been trained on a subset of
	# the training data. This gives us some of the training data that we can
	# use for validation.
	#
	# Unfortunately, by using a new classifier that is trained on less data
	# than the actual classifier, the metrics we print won't be totally
	# representative of the actual classifier. (Because we're using less
	# data to train the stat_classifier), we expect that the printed
	# metrics will provide a lower bound on the actual classifier's true
	# performance.
	if PRINT_METRICS:
		# Randomly split the training data into training and testing sets
		# for validation.
		train_in, test_in, train_out, test_out = \
			cross_validation.train_test_split(
				trainInput[:,1:], trainOutput, test_size=0.20)

		# Train a classifier on the train data and validate the results with the
		# testing set.
		# NOTE: The stats printed here aren't going to be 100% accurate, as we're
		# actually incorporating all of the data in our actual classification.
		stat_classifier = KNeighborsClassifier(n_neighbors = 30)
		# stat_classifier = GaussianNB()

		# Fit the classifier on the new training data and make predictions.
		stat_classifier.fit(train_in, train_out)
		predictions = stat_classifier.predict(test_in)

		# Print out the metrics by comparing the predictions to the actual
		# output.
		print "\n\nPurchased Last Viewed Classifier"
		print metrics.classification_report(test_out, predictions)
		print "Accuracy", metrics.accuracy_score(test_out, predictions)		

	# Iterate through the rows in the testing data. For those customers
	# predicted to purchase their last-shopped plan, construct the result
	# string (for the final output CSV) and tack it onto the end of
	# resultString. Also, we construct testIndicesToDelete such that it
	# contains all indices for which we predicted that the customer bought
	# their last-shopped plan.
	resultString = ""
	testIndicesToDelete = []
	for i in range(len(predictedTestOutputs)):
		if predictedTestOutputs[i] == True:
			resultString += str(int(testInput[i][0])) + ","
			for coverageOption in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
				resultString += \
					str(int(testInput[i][variableNames.index(coverageOption)]))
			resultString += '\n'
			testIndicesToDelete.append(i)

	# Iterate through the rows in the training data dn construct
	# trainIndicesToDelete such that it contains all indices for which we
	# predicted that the customer bought their last-shopped plan.
	trainIndicesToDelete = []
	for i in range(len(predictedTrainOutputs)):
		if predictedTrainOutputs[i] == True:
			trainIndicesToDelete.append(i)

	return resultString, trainIndicesToDelete, testIndicesToDelete

def secondClassifier(trainInput, purchasedOptions, testInput, variableNames):
	'''
	Classifier (actually 7 classifiers) to determine what each customer
	purchased for each of A-G. Returns a string containing the
	fully-formatted results.

	trainInput and testInput are the inputs for the training and testing sets.
	purchasedOptions is an array of 7-element arrays indicating what each
	customer purchased for each option.

	variableNames contains the names of the columns in trainInput and testInput.

	This function performs 7 classifications, one per option. After each
	classification, the value of the just-classified option in the input data
	(testInput and trainInput) is replaced with the predictions of the previous
	classifiers. Our hope in doing this was that giving each classifier the
	results of the classifiers that came before it would allow the classifiers
	to act upon correlations between columns.
	'''
	outputs = []

	# Iterate through the options and predict each.
	for i in range(len(['A', 'B', 'C', 'D', 'E', 'F', 'G'])):

		# Grab the output column from purchasedOptions.
		trainOutput = purchasedOptions[:,i]

		# Fit and train the classifier, then store the output from
		# classifying on both the training set and the testing set.
		# classifier = KNeighborsClassifier(n_neighbors = 30)
		# classifier = SVC()
		classifier = GaussianNB()
		classifier.fit(trainInput[:,1:], trainOutput)
		testPredictions = classifier.predict(testInput[:,1:])
		outputs.append(testPredictions)

		trainPredictions = classifier.predict(trainInput[:,1:])

		# Replace the values of the last-classified option in the feature
		# vectors with the just-predicted values.
		testInput[:,variableNames.index(
			['A', 'B', 'C', 'D', 'E', 'F', 'G'][i])] = testPredictions
		trainInput[:,variableNames.index(
			['A', 'B', 'C', 'D', 'E', 'F', 'G'][i])] = trainPredictions

		# If requested, print some metrics about the result of the
		# classification. We do this using a new classifier which has been
		# trained on a subset of the training data. This gives us some of
		# the training data that we can use for validation.
		#
		# Unfortunately, by using a new classifier that is trained on less
		# data than the actual classifier, the metrics we print won't be
		# totally representative of the actual classifier. (Because we're
		# using less data to train the stat_classifier), we expect that the
		# printed metrics will provide a lower bound on the actual classifier's
		# true performance.
		if PRINT_METRICS:
			# Randomly split the data into training and testing sets for validation.
			train_in, test_in, train_out, test_out = \
				cross_validation.train_test_split(
					trainInput[:,1:], trainOutput, test_size=0.20)

			# Train a classifier on the train data and validate the results with the
			# testing set.
			#stat_classifier = KNeighborsClassifier(n_neighbors = 30)
			stat_classifier = GaussianNB()
			stat_classifier.fit(train_in, train_out)
			predictions = stat_classifier.predict(test_in)

			# Print out the metrics by comparing the predictions to the actual
			# output.
			print "\n\nPlan", ['A', 'B', 'C', 'D', 'E', 'F', 'G'][i], "Classifier"
			print metrics.classification_report(test_out, predictions)
			print "Accuracy", metrics.accuracy_score(test_out, predictions)

	# After we've made our 7 classifications, determine the resultString (in
	# CSV format) for each customer.
	resultStrings = []
	customerIDs = testInput[:, variableNames.index('customer_ID')]
	for customerID, customerResult in zip(customerIDs, zip(*outputs)):
		resultString = str(int(customerID)) + ',' + \
			''.join([str(int(x)) for x in customerResult])
		resultStrings.append(resultString)

	# Return one large result string with the results for all customers.
	return '\n'.join(resultStrings)

def main():

	# Preprocess the training and testing sets.
	trainingCustomers, variableNames, purchasedLastViewed, purchasedOptions = \
		preprocess('train.csv', True)
	testCustomers, _, _, _ = preprocess('test_v2.csv', False)

	# Set up LabelEncoders to handle conversion of state and carValue values
	# from strings to ints.
	stateEncoder = preprocessing.LabelEncoder()
	carValueEncoder = preprocessing.LabelEncoder()

	# Fit the stateEncoder by stacking the corresponding train and
	# test columns and then fitting the encoder.
	#
	# By fitting the encoder on both the train data and the test data,
	# we can be sure that we won't encounter new values in the test data
	# on which the encoder will crash.
	#
	# This is probably a bit questionable from the perspective of never
	# incorporating the testing data in the training phase, but this should
	# have no effect on the actuall classification (for which we keep
	# the two sets completely separate).
	trainStates = trainingCustomers[:,variableNames.index('state')]
	testStates = testCustomers[:,variableNames.index('state')]
	stateStack = np.hstack((trainStates, testStates))
	stateEncoder.fit(stateStack)

	# Fit the carValueEncoder exactly as we trained the stateEncoder.
	trainCarValues = trainingCustomers[:,variableNames.index('car_value')]
	testCarValues = testCustomers[:,variableNames.index('car_value')]
	carValueStack = np.hstack((trainCarValues, testCarValues))
	carValueEncoder.fit(carValueStack)

	# Transform the training and testing columns for both states and
	# carValues using the fitted encoders.
	encodedTrainStates = stateEncoder.transform(trainStates)
	encodedTrainCarValues = carValueEncoder.transform(trainCarValues)
	encodedTestStates = stateEncoder.transform(testStates)
	encodedTestCarValues = carValueEncoder.transform(testCarValues)

	# Replace the original state and carValue columns with their
	# encoded equivalents.
	trainingCustomers[:,variableNames.index('state')] = encodedTrainStates
	trainingCustomers[:,variableNames.index('car_value')] = encodedTrainCarValues
	testCustomers[:,variableNames.index('state')] = encodedTestStates
	testCustomers[:,variableNames.index('car_value')] = encodedTestCarValues

	trainOutput = purchasedLastViewed
	trainInput = trainingCustomers
	testInput = testCustomers

	# Force NumPy to convert the values of the data (which are currently
	# strings parsed from the CSVs) into floats.
	trainInput = trainInput.astype(float)
	testInput = testInput.astype(float)
	trainOutput = trainOutput.astype(float)

	# Predict which customers purchased their last-shopped plan.
	resultString, trainIndicesToDelete, testIndicesToDelete = \
		lastViewedClassifier(trainInput, trainOutput, testInput, variableNames)

	# Delete from the data those customers we predicted had purchased their
	# last-shopped plan. This allows us to do the final classification on just
	# those who we predicted to NOT buy their last-shopped plan.
	trainInput = np.delete(trainInput, trainIndicesToDelete, axis=0)
	purchasedOptions = np.delete(purchasedOptions, trainIndicesToDelete, axis=0)
	testInput = np.delete(testInput, testIndicesToDelete, axis=0)

	# Perform the second classification to predict what each customer purchased
	# for each of A-G.
	resultString += \
		secondClassifier(trainInput, purchasedOptions, testInput, variableNames)

	# Write the header line and the resultString to the output file.
	# The resulting file will be suitable for submission to Kaggle.
	with open(OUTPUT_FILE_NAME, 'w') as f:
		f.write('customer_ID,plan\n')
		f.write(resultString)

if __name__ == "__main__":
	main()



