import sys, getopt

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "h:b:c", ["help","bridge_data=","component_data="])
	except getopt.GetoptError:
		print()
		print('Usage: brisk.py <bridgedatafile> <conditiondatafile>')
		sys.exit(2)
	for opt, arg in opts:			
		if opt in ('-h', '--help'):
			print('')
			print('Usage: brisk.py <bridgedatafile> <conditiondatafile>')
			sys.exit()
		elif opt == "--bridge_data":
			bridge_data = arg
		elif opt == "--component_data":
			component_data = arg
		else:
			print('')
			print('Usage: brisk.py <bridgedatafile> <conditiondatafile>')		
			
if __name__ == "__main__":
	main(sys.argv[1:])
	
import bridge