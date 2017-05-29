import os
import pandas as pd
import numpy as np
import datetime as dt
import time
from datetime import date
from sys import argv
from consequence import *

start = time.time()

def get_timestamp():
	t = dt.datetime.now()
	timestamp = t.strftime('%Y%m%d%I%M%p')
	return timestamp

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

path = os.path.dirname(os.path.abspath(__file__))

'''
Set argv to the list of command line arguments passed when running the script
bridge_data is the bridge level dataset
component_data is the component level dataset
'''
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
Declare global variables used throughout the script
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
	row['inspectionRating'] = row['inspectionRating']
	if row['inspectionRating'] == "":
		condition = 3
	else:
		condition = row['inspectionRating']
	row['constructionMaterial'] = row['constructionMaterial']
	if row['constructionMaterial'] == "":
		material = "Material unknown"
	else:
		material = row['constructionMaterial']
	row['inspectionDate'] = row['inspectionDate']
	if row['inspectionDate'] == "":
		inspectionDate = "uninspected"
	elif row['inspectionDate'] == "uninspected":
		inspectionDate = "uninspected"
	else:
		date = row['inspectionDate']
		inspectionDate = dt.datetime.strptime(date, '%d/%m/%Y')	
	if material == "Timber":
		ift = if_table[if_table['Overall condition state of structure'] == condition]
		inspection_interval = ift.ix['Timber', 'Inspection frequency (years)']
	else:
		ift = if_table[if_table['Overall condition state of structure'] == int(condition)]
		inspection_interval = ift.ix['Bridges and Culverts', 'Inspection frequency (years)']
	if inspectionDate ==  "uninspected":	
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
		if inspectionDate == "uninspected":
			pass
		else:
			inspectionDate = dt.datetime.strptime(inspectionDate, '%d/%m/%Y')
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

HC = hcon(df1)
TC = tcon(df1)
EC = econ(df1)
RS = rsig(df1)
AC = icon(df1)

def consequence_score(HC, TC, EC, RS, AC):
	results = []
	for a, b, c, d, e in zip(HC, TC, EC, RS, AC):
		try:
			result = a*b*c*d*e
			results.append(result)
		except (TypeError, ValueError):
			result = "Data missing"
			results.append(result)
	return results

CS = consequence_score(HC, TC, EC, RS, AC)
	
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

def mdfactor(dataframe1, dataframe2, *args): #dataframe1 is bridge data, dataframe2 is the condition data, dataframe3 is the updated condition data
	# Pass all calculation results into a list so a new column can be added to the dataframe with the existing index
	mdfresults = []
	dsrresults = []
	pgfresults = []
	bridgelist = []
	psfresults = []
	count = 0
	for index, row in dataframe1.iterrows():
		# Selects parentassetId from each row in the bridge data file
		bridgeAssetId = row['parentAssetId']
		# Selects all the rows in condition data where the parent asset id is equal to the bridgeAssetId variable
		conditionAssetId = dataframe2[dataframe2['parentAssetId'] == bridgeAssetId]
		# If no condition data exists for the parent asset id
		if conditionAssetId.empty:
			MDF = 1
			DSR = 1
			PSF = 1
			mdfresults.append(MDF)
			dsrresults.append(DSR)
			pgfresults.append('No data')
			bridgelist.append(bridgeAssetId)
			psfresults.append(PSF)
		# If condition data does exist, perform the MDF calculations
		else:
			# Pass the filtered dataframe to the group_mdf function
			filter_by_sr = conditionAssetId.loc[conditionAssetId['weighting'].isin([3,4])]
			groups = filter_by_sr.groupby('componentType').groups
			partial_mdf = []
			partial_dsr = []
			partial_pgf = []
			r = 0
			for key, values in groups.items():
				r += 1
				f_sum = 0
				pcf_sum = 0
				y_sum = 0
				x_sum = 0
				_3sum = 0
				_4sum = 0
				q = 0
				# Loops through all the components in a group, i.e will loop through S1 headstock, piles, etc
				for i in values:
					q += 1
					i_SF = SF[i]
					i_CF = CF[i]
					i_IF = IF[i]
					i_XF = XF[i]
					i_LF = LF[i]
					factor = (i_SF*i_CF*i_IF*i_XF)
					partial_pcf = i_SF*i_CF*i_IF*i_XF*i_LF
					pcf_sum += partial_pcf
					f_sum += factor
					# Error in here somewhere
					# Gets the index row from dataframe2
					df_row = dataframe2.iloc[i]
					# Gets the weighting of the component, can be 3 or 4
					i_weighting = df_row['weighting']
					# Get the measure of the component
					if pd.isnull(df_row['measure']):
						i_measure = 1
					else:
						i_measure = df_row['measure']
					# Get the amount of measure in CS4
					if pd.isnull(df_row['conditionStateFour']):
						i_no_cs4 = i_measure
					else:
						i_no_cs4 = df_row['conditionStateFour']
					# If the weighting was SR3
					if i_weighting == float(3):
						y = i_no_cs4/i_measure
						y_sum += y
						_3sum += 1
					# If the weighting was SR4
					elif i_weighting == float(4):
						x =  i_no_cs4/i_measure
						x_sum += x
						_4sum += 1
				#print('Factor sum is {0}'.format(f_sum))
				# Once loop is complete, will calculate the Partial Group Deficiency Factor
				mdfp = ((x_sum**m + 0.5*y_sum**m)/(_3sum + _4sum))*Fs
				# Calcualte the Probability of Group Failure
				PGF = (1+mdfp)*(pcf_sum/q)
				pgfresults.append(PGF)
				partial_pgf.append(PGF)
				bridgelist.append(bridgeAssetId)
				# Once loop is complete, will calculate the Partial Group Defective Structure Rating
				dsrg = ((1+mdfp)*(f_sum))/(q)
				# Adds the partial group MDF to a list, so for each structure the sum of groups can be obtained
				partial_mdf.append(mdfp)
				# Adds the partial group DSR to a list, so for each structure the sum of groups can be obtained
				partial_dsr.append(dsrg)
			# Calculates the total structure MDF by summing the partial groups
			MDF = 1 + sum(partial_mdf)
			# Calculates the total structure DSR by summing the partial groups
			DSR = ((sum(partial_dsr))/r)*(MDF/Fc)
			# Calcualtes the probability of structure failure
			PSF = (((sum(partial_pgf))/(1+sum(partial_mdf)))/(len(pgfresults)))*((MDF)/(Fc))
			# Adds the MDF for the structure to a list
			mdfresults.append(MDF)
			#print('Structure MDF is {0}'.format(MDF))
			dsrresults.append(DSR)
			# Adds the PSF for the structure to a list
			psfresults.append(PSF)
		#except:
		#count += 1
		#print('Error {0}'.format(count))
	return mdfresults, dsrresults, pgfresults, bridgelist, psfresults

mdfresults, dsrresults, pgfresults, bridgelist, psfresults = mdfactor(df1, df2, SF, CF, IF, XF, LF)	

def risk_score(pof, cof):
	results = []
	for a, b in zip(pof, cof):
		try:
			result = a*b
		except:
			pass
		results.append(result)
	return results
	
risk = risk_score(psfresults, CS)
	
def output_group_calculations(pgf_list, bridge_index):
	df = pd.Series(pgf_list, index=bridge_index)
	timestamp = get_timestamp()
	output = df.to_csv(os.path.join(path, timestamp + 'group.csv'))
	return output
	
output_group_calculations(pgfresults, bridgelist)

def output_bridge_calculations(dataframe, bridge_index, *args):
	dataframe['replacementValue'] = dataframe.apply (lambda row: add_replacement_cost (row), axis=1)
	dataframe['nextInspection'] = dataframe.apply (lambda row: next_inspection_date (row, inspection_frequency), axis=1)
	index= ['MDF', 'DSR', 'PSF', 'COF', 'Risk']
	dataframe2 = {}
	i = 0
	for arg in args:
		#print(arg)
		dataframe2[index[i]] = pd.Series(arg, index=dataframe.index)
		i += 1
	df = pd.DataFrame(dataframe2)
	return df
	
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
	return df

def update_bridge_data(dataframe1, dataframe2):
	new_df = pd.concat([dataframe1, dataframe2], axis=1)
	timestamp = get_timestamp()
	output = new_df.to_csv(os.path.join(path, timestamp + 'bridge' + '.csv'))
	return output


def update_condition_data(dataframe1, dataframe2):
	new_df = pd.concat([dataframe1, dataframe2], axis=1)
	timestamp = get_timestamp()
	output = new_df.to_csv(os.path.join(path, timestamp + 'components' + '.csv'))
	return output
	
df3 = output_component_calculations(component_index, LF, SF, CF, IF, XF, PCF)
df4 = output_bridge_calculations(df1, bridge_index, mdfresults, dsrresults, psfresults, CS, risk)
#output_bridge_inspection_list(dueForInspectionList, detailedDueForInspectionList)
update_condition_data(df2, df3)
update_bridge_data(df1, df4)

def on_script_end():
	print("\n")
	print("... Analysis complete.")
	print('... It took {0:0.1f} seconds to run the analysis.'.format(time.time() - start))
	print("\n")

on_script_end()	