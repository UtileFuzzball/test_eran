import csv

with open('','w',newline='')as csvfile:
	csvreader=csv.reader(csvfile,delimier=',')
	for row in spamreader:
		print(row)