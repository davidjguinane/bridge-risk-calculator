import pandas as pd

rc_rate = 7000.0
# Human Consequence Factor

def hcon(dataframe):
	results = []
	for index, row in dataframe.iterrows():
		aadt = row['aadt']
		if aadt < 1000:
			HF = 1
		elif aadt >= 1000 and  aadt < 5000:
			HF = 2
		elif aadt >= 5000 and  aadt < 10000:
			HF = 3
		elif aadt >= 10000 and  aadt < 50000:
			HF = 4
		elif aadt >= 50000:
			HF = 5
		results.append(HF)
	return results
	
# Environmental Consequence Factor

# Traffic Access Consequence Factor

def tcon(dataframe):
	results = []
	for index, row in dataframe.iterrows():
		detour = row['detourLength']
		if detour < 2:
			TC = 0.5
		elif detour >= 2 and  detour < 5:
			TC = 1
		elif detour >= 5 and  detour < 10:
			TC = 1.5
		elif detour >= 10:
			TC = 2
		elif detour == None:
			TC = 3
		results.append(TC)
	return results

# Economic Consequence Factor

def econ(dataframe):
	results = []
	for index, row in dataframe.iterrows():
		area = row['area']
		rCost = area * rc_rate
		#print(rCost)
		if rCost < 73430:
			EF = 1
		elif rCost >= 73430 and  rCost < 293706:
			EF = 2
		elif rCost >= 293706 and  rCost < 734266:
			EF = 3
		elif rCost >= 734266 and  rCost < 1468533:
			EF = 4
		elif rCost >= 1468533:
			EF = 5
		results.append(EF)
	return results

# Road Significance Factor

def rsig(dataframe):
	results = []
	for index, row in dataframe.iterrows():
		hierarchy = row['roadHierarchy']
		if hierarchy == '5U':
			RS = 5
		elif hierarchy == '5R':
			RS = 5
		elif hierarchy == '4U':
			RS = 4
		elif hierarchy == '4R':
			RS = 4
		elif hierarchy == '3U':
			RS = 3
		elif hierarchy == '3R':
			RS = 3
		elif hierarchy == '2U':
			RS = 2
		elif hierarchy == '2R':
			RS = 2
		elif hierarchy == '1U':
			RS = 1
		elif hierarchy == '1R':
			RS = 1
		results.append(RS)
	return results

# Local Industry Access Consequence Factor

def icon(dataframe):
	results = []
	for index, row in dataframe.iterrows():
		bool = row['industryDependent']
		if bool == True:
			AC = 1.5
		elif bool == False:
			AC = 0
		results.append(AC)
	return results

# Consequence of Failure Calculation

# Pass cfailure all consequence factors for addition
def cfailure(*args):
	COF = 0
	for arg in args:
		COF += arg
	return COF

# Pass orequrie Human Consequence Factor, Traffic Consequence Factor & Road Significance Factore  	
def orequire(*args):
	orequire = 0
	for arg in args:
		orequire += arg
	return orequire
	
def risk(pof, cof):
	return pof*cof
	
def oimport(pof, orequire):	
	return pof*orequire





