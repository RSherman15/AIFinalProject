#CS 151 Final Project
#Emily Blatter, Michael Culhane, and Rachel Sherman

import csv
import time
import datetime
import math
from collections import Counter
import numpy as np

def processCustomer(rows, variableNames):
	processedRow = []

	for i in range(len(variableNames)):
		if variableNames[i] == 'time':
			times = []
			for row in rows:
				x = time.strptime(row[i], "%H:%M")
				seconds = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
				times.append(seconds)
			averageTime = sum(times)/float(len(times))
			processedRow.append(averageTime)
		
		elif variableNames[i] == 'cost':
			costs = []
			for row in rows:
				costs.append(int(row[i]))
			averageCost = sum(costs)/float(len(costs))
			processedRow.append(averageCost)
		
		elif variableNames[i] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
			options = []
			for row in rows:
				options.append(row[i])
			counter = Counter(options)
			mode, _ = counter.most_common(1)[0]
			processedRow.append(int(mode))

		elif variableNames[i] in ['car_value', 'state']:
			processedRow.append(rows[0][i])

		else:
			if rows[0][i] == 'NA':
				processedRow.append(-1)
			else: 
				processedRow.append(int(rows[0][i]))

	return processedRow



def main():
	with open('train.csv', 'r') as f:
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

	customers = np.array(customers)
	output = np.delete(customers, variableNames.index('A'), 1)
	print output

if __name__ == "__main__":
	main()



