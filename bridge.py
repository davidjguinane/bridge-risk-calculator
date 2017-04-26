import os
import pandas as pd
import numpy as np
import datetime as dt
import time
from datetime import date
from sys import argv

start = time.time()

def add_years(d, years):
    """Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).
    """
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))

path = 'C:\\Users\\DJG\\Desktop\\scripts'

# set argv to the list of arguments passed to the script
script, bridge_data, component_data = argv

def on_script_run():
	print("\n")
	print("... BRIDGE AND CULVERT MAINTENANCE PRIORITISATION SOFTWARE")
	print("... Version == 0.9 beta")
	print("... Copyright 2017")
	print("... Written by David Guinane")
	print("\n")

on_script_run()	
	
''' 
***DECLARE SCRIPT CONSTANTS / TABLES***
'''

Fs = 10 # Seperation Factor
m = 2 # From DTMR Manual
Fc = 1.84 # Calibration Factor
rc_rate = 7000.0 # Assumed Replacement Cost Unit Rate


"""Returns the replacement cost of the bridge in 
a dataframe column by multipying the global unit 
rate variable by the area of the bridge.
"""
def add_replacement_cost(row):
	area = row['area']
	try:
		replacement_cost = rc_rate*float(area)
		return replacement_cost
	except (TypeError, ValueError):
		replacement_cost = "Data not a float"
		return replacement_cost
		
"""Returns the date in DD/MM/YYYY format
of the next inspection date of the bridge structure.
"""
def next_inspection_date(row, if_table):
	current_date = dt.datetime.today()
	try:
		row['inspectionRating'] = row['inspectionRating'].fillna(3.0, inplace=True)
		condition = row['inspectionRating']
	except (AttributeError, TypeError, ValueError):
		condition = 3
	#print(condition)
	try:
		row['constructionMaterial'] = row['constructionMaterial'].fillna("unknown", inplace=True)
		material = row['constructionMaterial']
	except (AttributeError, TypeError, ValueError):
		material = "Material unknown"
	try:
		row['inspectionDate'] = row['inspectionDate'].fillna("uninspected", inplace=True)
		date = row['inspectionDate']
	except (AttributeError, TypeError, ValueError):
		date = "uninspected"
	if material == "Timber":
		ift = if_table[if_table['Overall condition state of structure'] == condition]
		inspection_interval = ift.ix['Timber', 'Inspection frequency (years)']
	else:
		ift = if_table[if_table['Overall condition state of structure'] == int(condition)]
		inspection_interval = ift.ix['Bridges and Culverts', 'Inspection frequency (years)']
	if date ==  "uninspected":	
		nextInspectionDate = current_date
	elif add_years(inspectionDate, inspection_interval) < current_date:
		nextInspectionDate = current_date
	else:
		nextInspectionDate = add_years(inspectionDate, inspection_interval)
	return dt.datetime.strftime(nextInspectionDate, '%d/%m/%Y')

"""Creates Table 1A from the TMR Prioritisation User Guide
as a dataframe 
"""	
def create_table_1a():
	index1 = ['BM','B','AM','A','H20S16','T44','T44+HLP','SM1600']
	data_1 = {
		'100' : pd.Series([8,6,5,4,3,2,1,1], index=index1),
		'500' : pd.Series([12,9,6,5,4,3,2,1], index=index1),
		'1000' : pd.Series([16,12,8,6,5,4,3,1], index=index1),
		'5000' : pd.Series([20,15,10,8,7,5,4,2], index=index1),
		'10000000' : pd.Series([24,18,12,10,8,6,5,3], index=index1)
	}
	table1A = pd.DataFrame(data_1)
	return table1A
	
table1A = create_table_1a()

"""Creates Table 1B from the TMR Prioritisation User Guide
as a dataframe 
"""
def create_table_1b():	
	index2 = ['Up to 1925 (BM)','1925-1935 (B)','1935 - 1945 (AM)','1945 - 1955 (A)','1955 - 1975 (H20S16)','1975 - 192 (T44)','1992 - 202 (T44+HLP)','2002+ (SM1600)']
	data_2 = {
		'100' : pd.Series([8,6,5,4,3,2,1,1], index=index2),
		'500' : pd.Series([12,9,6,5,4,3,2,1], index=index2),
		'1000' : pd.Series([16,12,8,6,5,4,3,1], index=index2),
		'5000' : pd.Series([20,15,10,8,7,5,4,2], index=index2),
		'10000000' : pd.Series([24,18,12,10,8,6,5,3], index=index2)
	}
	table1B = pd.DataFrame(data_2)
	return table1B
	
table1B = create_table_1b()

"""Creates Exposure Factor Parameter table from the TMR Prioritisation User Guide
as a dataframe 
"""
def create_exposure_table():
	index = ['Relatively benign','Mildly aggresive','Aggresive','Most aggressive']
	exposure = {'Classification' : pd.Series([1,2,3,4], index=index), 'XF' : pd.Series([1,1.41,1.73,2], index=index)}
	table4 = pd.DataFrame(exposure)
	return table4
	
xf = create_exposure_table()

"""Creates Resistance Factor Parameter table from the TMR Prioritisation User Guide
as a dataframe 
"""
'''
def create_resistance_table():		
	componentNo = [1,2,3,4,10,11,12,13,14,15,20,21,22,23,24,25,26,27,28,29,30,31,32,33,40,41,42,43,44,45,50,51,52,53,54,55,56,57,58,59,70,71,72,80,81,82,83,84]
	precast=[2,3,4,20,21,22,25,29,52,53,54,56,72,80,81,82,83,84]
	concrete=[1,2,3,4,20,21,22,25,27,31,32,50,51,52,53,54,55,56,57,58,59,71,72,81,83,84]
	steel = [2,3,4,13,14,21,22,23,24,25,26,28,30,31,43,45,52,54,56,57,72,80,83]
	timber = [2,3,4,20,22,27,28,29,32,52,54,56,5,59,72]
	other = [1,2,4,10,11,12,13,15,25,40,41,42,44,45,50,51,52,53,58,70,71,72,80,83,84]
	resistance_factors = {'Description' : pd.Series(['Wearing surface','Bridge Barriers', 'Bridge Kerbs', 'Footways', 'Pourable Joint Seal', 'Compression Joint Seal', 'Assembly Joint Seal', 'Open Expansion Joint', 'Sliding Joint', 'Fixed/Small Movement Joint', 'Deck Slab/ Culvert Base Slab Joints', 'Closed Web/Box Girders', 'Open Girders', 'Through Truss', 'Deck Truss', 'Arches', 'Cables/Hanger', 'Corbels', 'Cross Beams/Floor Beams', 'Deck Planks', 'Steel Decking', 'Diaphragms/Bracing (Cross Girders)', 'Load Bearing Diaphragm', 'Spiking Plank', 'Fixed Bearings', 'Sliding Bearings', 'Elastomeric/Pot Bearings', 'Rockers/Rollers', 'Mortar Pads/Bearing Pedestals', 'Restraint Angles/Blocks', 'Abutment', 'Wingwall/Retaining Wall', 'Abutment Sheeting/Infill Panels', 'Batter Protection', 'Headstocks', 'Pier Headstocks (Integral)', 'Columns of Piles', 'Piles Bracing/Walls', 'Pier Walls', 'Footing/Pile Cap/Sill Log', 'Bridge Approaches', 'Waterway', 'Approach Guardrail', 'Pipe Culverts', 'Box Culverts', 'Modular Culverts', 'Arch Culverts', 'Headwalls/Wingwalls'], index=componentNo),
	'Significance Rating (SR)' : pd.Series([2,1,1,1,2,2,2,2,2,2,3,4,4,4,4,4,4,3,3,3,3,3,4,1,2,2,2,2,1,2,3,3,2,1,4,4,4,3,3,3,2,2,1,2,2,2,2,1], index=componentNo), 
	'Precast Concrete' : pd.Series([1,1,1,3,4,4,4,3,2,1,4,4,1,2,2,2,2,1], index=precast),
	'Concrete' : pd.Series([4,2,2,2,6,8,8,8,6,6,8,6,6,4,2,8,8,8,6,6,6,4,2,4,4,2], index=concrete),
	'Steel' : pd.Series([3,3,3,6,6,12,12,12,12,12,12,9,9,9,6,6,6,12,12,9,3,6,6], index=steel),
	'Timber' : pd.Series([4,4,4,12,16,12,12,12,4,8,16,16,12,12,4], index=timber),
	'Other' : pd.Series([3,1.5,1.5,3,3,3,3,3,6,3,3,3,1.5,3,4.5,4.5,3,1.5,4.5,3,3,1.5,3,3,1.5], index=other)}
	table2 = pd.DataFrame(resistance_factors)
	return table2

table2 = create_resistance_table()	
'''	
"""Opens Resistance Factor Parameter table a CSV file
"""
def read_table_2(data):
	table2 = pd.read_csv(data)
	#print(table2)
	return table2
	
table2 = read_table_2('table2.csv')

"""Creates Inspection Frequency Table from the TMR SIM
as a dataframe 
"""
def create_inspection_frequency_table():
	index = ['Timber','Timber','Timber','Timber','Bridges and Culverts','Bridges and Culverts','Bridges and Culverts','Bridges and Culverts']
	inspection_frequency = {
		'Overall condition state of structure' : pd.Series([1,2,3,4,1,2,3,4], index=index),
		'Inspection frequency (years)' : pd.Series([2,2,1,1,5,5,3,1], index=index),
	}
	frequency_table = pd.DataFrame(inspection_frequency)
	return frequency_table	

inspection_frequency = create_inspection_frequency_table()
	
'''
*** READ THE DATA ***
'''

# Read the Bridge Level Data CSV file to a Pandas Dataframe 
def read_bridge_data():
	# Read the bridge CSV
	bridge_df = pd.read_csv(bridge_data, parse_dates=True, dayfirst=True)
	return bridge_df	
	
# Read the Component Level Data CSV file to a Pandas Dataframe 
def read_component_data():
	# Read the bridge CSV
	component_df = pd.read_csv(component_data, parse_dates=True, dayfirst=True)
	return component_df

df1 = read_bridge_data()	
df2 = read_component_data()

component_index = df2.index.tolist()
bridge_index = df1.index.tolist()

'''
*** CHECK DATA TYPES ***
'''

def check_data_types():
	print("\n")
	print("... Your bridge data is in the following format:")
	print(df1.dtypes)
	print("\n")
	print("... Your component data is in the following format:")
	print(df2.dtypes)
	print("\n")

#check_data_types()
	
def get_design_class(dataframe):
	design_class = dataframe['designClass']
	return design_class
	
def get_cvpd(dataframe):
	cvpd= dataframe['cvpd']
	result = cvpd.fillna(cvpd.mean())
	return result

'''
*** LOADING FACTOR ***
'''
	
def loading_factor(dataframe, table1a):
	# iterate through dataframe
	results = []
	bridgeAssetList = []
	
	for index, row in dataframe.iterrows():
			result = table1a.ix['T44', '500']
			results.append(result)
	return results

LF = loading_factor(df2, table1A)

'''
*** RESISTANCE FACTOR ***
'''

# Calculate the Resistance Factor in accordance with TMR Manual Section 5.2
def resistance_factors(dataframe1, dataframe2): #dataframe1 is component data, dataframe2 is table2
	results = []
	# Get the asset ID from the line in dataframe1 - component data
	for index, row in dataframe1.iterrows():
		# select the material
		material = row['material']
		# Select compoent
		component = row['subComponent']
		# Set dataframe index to Component description
		df = dataframe2.set_index('Description')
		# Indexes the dataframe for a matching description to component and returns the corresponding SR value
		SR = df.ix[component, 'Significance Rating (SR)']
		# Indexes the dataframe for a matching description to component and returns the corresponding material significance value
		MW = df.ix[component, material]
		# Returns the product of the SR and material
		result = SR*MW
		results.append(result)	
	# Get Asset ID Index and assign to index variable
	#index = dataframe1.index.tolist()	
	# Turn result data into a Dataframe
	#df = pd.DataFrame(results, index=index)
	return results
	
SF = resistance_factors(df2, table2)

'''
*** CONDITION FACTOR ***
'''

def condition_fator(dataframe):
	results = []
	# Get the asset ID from the line in dataframe - component data
	for index, row in dataframe.iterrows():
		measure = row['measure']
		pd.to_numeric(measure)
		condition_state_one = row['conditionStateOne']
		condition_state_two = row['conditionStateTwo']
		condition_state_three = row['conditionStateThree']
		condition_state_four = row['conditionStateFour']
		condition_state_five = row['conditionStateFive']
		perc_state_one = condition_state_one / measure
		perc_state_two = condition_state_two / measure
		perc_state_three = condition_state_three / measure
		perc_state_four = condition_state_four / measure
		perc_state_five = condition_state_five / measure
		result = (1*perc_state_one) + (2*perc_state_two) + (3*perc_state_three) + (4*perc_state_four) + (5*perc_state_five)
		results.append(result)
	return results
	
CF = condition_fator(df2)

'''
*** INSPECTION FACTOR ***
'''

def check_inspection_interval_exceeded(dataframe, inspection_frequency_table):
	
	current_date = dt.datetime.today()
	# Get all columns needed, Fill empty cells with defaults
	dataframe['inspectionRating'].fillna(3.0, inplace=True)
	dataframe['constructionMaterial'].fillna("unknown", inplace=True)
	dataframe['inspectionDate'].fillna("uninspected", inplace=True)
	results = []
	dueForInspectionList = []
	detailedDueForInspectionList = []
	
	for index, row in dataframe.iterrows():
		bridgeStreet = row['street']
		bridgeName = row['bridgeName']
		assetId = row['parentAssetId']
		inspectionRating = row['inspectionRating']
		constructionMaterial = row['constructionMaterial']
		inspectionDate = row['inspectionDate']
		#print(inspectionDate)
		if inspectionDate == "uninspected":
			pass
		else:
			inspectionDate = dt.datetime.strptime(inspectionDate, '%d/%m/%Y')
		#print(inspectionDate)
		if constructionMaterial == "Timber":
			ift = inspection_frequency_table[inspection_frequency_table['Overall condition state of structure'] == inspectionRating]
			inspection_interval = ift.ix['Timber', 'Inspection frequency (years)']
		else:
			ift = inspection_frequency_table[inspection_frequency_table['Overall condition state of structure'] == inspectionRating]
			inspection_interval = ift.ix['Bridges and Culverts', 'Inspection frequency (years)']
		if inspectionDate ==  "uninspected":
			result = assetId
			dueForInspectionList.append(result)
			detailedResult = (bridgeStreet, bridgeName)
			detailedDueForInspectionList.append(detailedResult)
		elif add_years(inspectionDate, inspection_interval) < current_date:
			result = assetId
			dueForInspectionList.append(result)
			detailedResult = (bridgeStreet, bridgeName)
			detailedDueForInspectionList.append(detailedResult)
		else:
			result = assetId
			pass
	#print(dueForInspectionList)
	#print(len(dueForInspectionList))
	return dueForInspectionList, detailedDueForInspectionList
	
dueForInspectionList, detailedDueForInspectionList = check_inspection_interval_exceeded(df1, inspection_frequency)
			
def inspection_factor(dataframe1, dataframe2, inspection_frequency_table): #dataframe1 is condition data, dataframe2 is bridge level data
	
	current_date = dt.datetime.today().strftime("%m/%d/%Y")
	lessThan25Inspected = dataframe1['lessThanTwentyFivePercentInspected']
	lessThan25Inspected.fillna("Yes")
	results = []
	
	dueForInspectionList, detailedDueForInspectionList = check_inspection_interval_exceeded(dataframe2, inspection_frequency_table)
	
	for index, row in dataframe1.iterrows():
		assetId = row['parentAssetId']
		lessThan25Inspected = row['lessThanTwentyFivePercentInspected']
		if lessThan25Inspected == "No":
			lessthan25 = False
		else:
			lessthan25 = True
		if lessthan25 or assetId in dueForInspectionList is True:
			result = 2
		else:
			result = 1
		results.append(result)	
	return results
	
IF = inspection_factor(df2, df1, inspection_frequency)

'''
*** EXPOSURE FACTOR ***
'''

def exposure_factor(dataframe, exposure_table): #dataframe is component data
	
	results = []

	for index, row in dataframe.iterrows():
		exposure_classification = row['exposureClass']
		df = exposure_table.set_index('Classification')
		result = df.ix[exposure_classification, 'XF']
		results.append(result)
	return results
		
XF = 	exposure_factor(df2, xf)

def probability_component_failure(LF, SF, CF, IF, XF):
	results = []
	for a, b, c, d, e in zip(LF, SF, CF, IF, XF):
		try:
			result = a*b*c*d*e
			results.append(result)
		except (TypeError, ValueError):
			result = "Data missing"
			results.append(result)
	return results
	
PCF = probability_component_failure(LF, SF, CF, IF, XF)

def group_mdf(condition_data, dataframe):
	#print(condition_data)
	filter_by_sr = condition_data.loc[condition_data['weighting'].isin([3,4])]
	#print(filter_by_sr)
	groups = filter_by_sr.groupby(['componentType', 'componentGroup']).groups
	#print(groups)
	mdf = []
	group_mdf = []
	for key, values in groups.items():
		#print(key)
		#print(values)
		y_sum = 0
		x_sum = 0
		_3sum = 0
		_4sum = 0
		for i in values:
			#print(i)
			df_row = dataframe.iloc[i]
			#print(df_row)
			i_weighting = df_row['weighting']
			#print("working")
			#print('The weighting of the component is ' + i_weighting)
			i_measure = df_row.loc[i,'measure']
			#print('The measure of the component is ' + i_measure)
			i_no_cs4 = df_row.loc[i,'conditionStateFour']
			#print('The measure of the component that is in CS4 is ' + i_no_cs4)
			if i_weighting == float(3):
				y = i_no_cs4/i_measure
				y_sum += y
				_3sum += 1
				#print(y)
			elif i_weighting == float(4):
				x =  i_no_cs4/i_measure
				x_sum += x
				_4sum += 1
				#print(x)
			#print('The sum of x is ' + x_sum)
			#print('The sum of y is ' + y_sum)
			#print('The sum of xt is ' + _3sum)
			#print('The sum of yt is ' + _4sum)
		mdfp = ((x_sum**m + 0.5*y_sum**m)/(_3sum + _4sum))*Fs
		mdf.append(mdfp)
		group_mdf.append(mdfp)
	result = 0
	for i in mdf:
		result += i
	return result, group_mdf
		
def bridge_mdf(dataframe1, dataframe2): #dataframe1 is bridge data
	results = []
	group_mdf_results = []
	for index, row in dataframe1.iterrows():
		bridgeAssetId = row['parentAssetId']
		try:
			conditionAssetId = dataframe2[dataframe2['parentAssetId'] == bridgeAssetId]
			result, group_mdf = group_mdf(conditionAssetId, dataframe2)
			results.append(result)
			group_mdf_results.append(group_mdf)
		except:
			result = 1
			results.append(result)
	return results

MDF = bridge_mdf(df1, df2)

'''
*** DEFECTIVE STRUCTURE RATING ***
'''

def group_dsr():
	pass
	
def dsrating():
	pass

'''
*** PROBABILITY OF GROUP FAILURE ***
'''
'''
def output_group_calculations(*args):
	dataframe = {}
	group_index = ()
	count = 0
	for arg in args:
		#print(arg)
		dataframe[count] = pd.Series(arg, index=index)
		count += 1
	df = pd.DataFrame(dataframe)
	output = df.to_csv(filename)
	return output
'''
	
def prob_group_failure(mdfp):
	pass
	
#PGF = prob_group_failure

'''
*** PROBABILITY OF STRUCTURE FAILURE ***
'''
	
def prob_structure_failure():
	results = []
	
	return results

#PSF = prob_structure_failure
	
'''
*** OUTPUT THE DATA ***
'''
'''
def output_bridge_calculations(bridge_index, *args):
	datetime = dt.datetime.today().strftime('%Y-%m-%d')
	filename = str(datetime) + "bridge"
	dataframe = {}
	index= ['RV','MDF']
	i = 0
	for arg in args:
		#print(arg)
		dataframe[index[i]] = pd.Series(arg, index=bridge_index)
		i += 1
	df = pd.DataFrame(dataframe)
	output = df.to_csv(filename)
	return output, filename
'''
	
# Save the calculated data to an output CSV
def output_component_calculations(component_index, *args):
	datetime = dt.datetime.today().strftime('%Y-%m-%d')
	filename = str(datetime) + "component"
	dataframe = {}
	index= ['LF','SF','CF','IF','XF', 'PCF']
	i = 0
	for arg in args:
		#print(arg)
		dataframe[index[i]] = pd.Series(arg, index=component_index)
		i += 1
	df = pd.DataFrame(dataframe)
	#output = df.to_csv(os.path.join(path, filename + '.csv'))
	#return output, filename
	return df

'''
def output_bridge_inspection_list(list, detailedList):
	datetime = dt.datetime.today().strftime('%Y-%m-%d')
	filename = str(datetime) + "inspections"
	dataframe['Inspections'] = pd.Series(detailedList, index=list)
	df = pd.DataFrame(dataframe)
	output = df.to_csv(os.path.join(path, filename + '.csv'))
	return output, filename
'''	
def update_bridge_data(dataframe, bridge_index, *args):
	dataframe['replacementValue'] = dataframe.apply (lambda row: add_replacement_cost (row), axis=1)
	dataframe['nextInspection'] = dataframe.apply (lambda row: next_inspection_date (row, inspection_frequency), axis=1)
	index= ['MDF']
	i = 0
	for arg in args:
		#print(arg)
		dataframe[index[i]] = pd.Series(arg, index=bridge_index)
		i += 1
	new_df = pd.DataFrame(dataframe)
	updated_df = pd.concat([dataframe, new_df], axis=1)
	output = updated_df.to_csv(os.path.join(path, 'bridge-level-edited' + '.csv'))
	return output	
	
update_bridge_data(df1, bridge_index, MDF)

def update_condition_data(dataframe1, dataframe2):
	new_df = pd.concat([dataframe1, dataframe2], axis=1)
	output = new_df.to_csv(os.path.join(path, 'condition-edited' + '.csv'))
	return output
	
#output_filename = output_bridge_calculations(bridge_index, MDF)
df3 = output_component_calculations(component_index, LF, SF, CF, IF, XF, PCF)
#output_bridge_inspection_list(dueForInspectionList, detailedDueForInspectionList)
update_condition_data(df2, df3)

def on_script_end():
	print("\n")
	print("... Analysis complete.")
	print('... It took {0:0.1f} seconds to run the analysis.'.format(time.time() - start))
	#print("... The bridge ouptut data file is named %s" %  bridge_output_filename)
	#print("... The component ouptut data file is named {0}".format(component_output_filename))
	print("\n")

on_script_end()	