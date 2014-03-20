import sys, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

""" 

This program was written for the TagMe contest conducted by IISc Banglore
What: Image Tagging 
Duration: 1/03/2014 to 14/03/2014
Username in the contest: Se7en (Rank 9 on validation test)
In this I used two types of feature extraction
1) Features provided in contest
2) Features extracted using OverFeat (see in google)
Two classifiers were tested
1) RandomForest in Sklearn
2) SVM in Sklearn
In the end I didn't submitted the test set :D

"""

# Pre-processing step: scales the data based on noraml distribution
def pre_process_data(data):
	scaler = StandardScaler()
	data = scaler.fit_transform( data )
	return data


def read_train_data_dir(feature_dir, label_file):
        file_list = os.listdir(feature_dir)
        labels_file_object = open(label_file,'r') # read the labels file

        labels = {} # create labels dictionary
        dim = 4096
    
        for label in labels_file_object.readlines():
                img_name, tag = label.strip().split() # it strips the line and splits based space
                labels[img_name] = int(tag)  # assign each image with their corresponding tag
    
        train_data = []
        target = []
        for file in file_list:
                file_name, ext = os.path.splitext(file) # spliting the file by extension
                file = os.path.join(feature_dir, file)  # join the full path of the file
                data = np.loadtxt(file, skiprows = 1)   # read the file in txt format with head skip
                if data.size > dim: # if it is not square image then it will give ndim feature 
                        # max pooling  is one method to get 1 dim data
                        ndim = data.size / dim          # finding the n dim  
                        data = np.reshape( data, dims )             # 1D to ND
                        data = np.amax( np.amax( data, 1 ), 1 )     # max over columns and rows
    
                train_data.append(data) # appending the data into train_data
                target.append(labels[file_name]) # appending the target 
   	
	train_data = np.array(train_data) 
        return train_data, target	

def read_train_data(train_file, label_file):
	###### read training data #######

	txt_file_object = open(train_file,'r') # read the features file
	labels_file_object = open(label_file,'r') # read the labels file

	labels = {} # create labels dictionary

	for label in labels_file_object.readlines():
		img_name, tag = label.strip().split() # it strips the line and splits based space
		labels[img_name] = tag  # assign each image with their corresponding tag


	train_data=[] #Creat a variable called 'train_data'
	target = [] # Creat a variable called 'target'
	for row in txt_file_object.readlines():
		parts = row.strip().split() #strip and split the line based on spaces
		target.append(labels[parts[0]]) # replace the filenames with tags
		train_data.append(parts[1:]) # append the rows into train_data

	train_data = np.array(train_data) #Then convert from a list to an array
	return train_data, target


def read_test_data_dir(test_dir):
	###### read validataion data ########

	file_list = os.listdir(test_dir) # open validatation file
	image_ids = [] # create a empty list for image id
	test_data = [] # create a empty list for test data
	dim = 4096
	for file in file_list:
		file_name, ext = os.path.splitext(file)
		file = os.path.join(test_dir,file)
		data = np.loadtxt(file, skiprows = 1)   # read the file in txt format with head skip
                if data.size > dim: # if it is not square image then it will give ndim feature 
                        # max pooling  is one method to get 1 dim data
                        ndim = data.size / dim          # finding the n dim  
                        data = np.reshape( data, dims )             # 1D to ND
                        data = np.amax( np.amax( data, 1 ), 1 )     # max over columns and rows

		image_ids.append(file_name)
    		test_data.append(data) #adding each row to the data variable
	
	test_data = np.array(test_data)
	return image_ids, test_data

def read_test_data(test_file):
        ###### read validataion data ########

        test_file_object = open(test_file,'r') # open validatation file
        image_ids = [] # create a empty list for image id
        test_data = [] # create a empty list for test data

        for row in test_file_object:
                parts = row.strip().split() #strip and split the line based on spaces    
                image_ids.append(parts[0])
                test_data.append(parts[1:]) #adding each row to the data variable

        test_data = np.array(test_data) #Then convert from a list to an array
        return image_ids, test_data


def RandomForest_method(train_data, target, test_data, n_est):
	#Use Random Forest method for training and testing!
	print 'Forest Training'
	forest = RandomForestClassifier(n_estimators = n_est, criterion='gini')

	forest = forest.fit(train_data, target)


	print 'Forest Predicting'
	output = forest.predict(test_data)
	return output

def SVM_method(train_data, target, test_data):
	# Create a classifier: a support vector classifier
	classifier = svm.SVC(gamma=0.001, kernel='poly', degree = 4)
	
	print 'SVM training'
	# We learn the digits on the first half of the digits
	classifier.fit(train_data, target)
	
	print 'SVM predicting'
	# Now predict the value of the digit on the second half:
	output = classifier.predict(test_data)
	return output

def main():
	if len(sys.argv) != 5:
		print "Error..! \n"
		print "Usage: python tagme.py label_file train_file/train_dir test_file/test_dir output_file"
		sys.exit(2)
	else:
		label_file = sys.argv[1]
		
		if os.path.isfile(sys.argv[2]):
			train_file = sys.argv[2]
			train_data, target = read_train_data(train_file, label_file)
		else:
			train_dir = sys.argv[2]
			train_data, target = read_train_data_dir(train_dir, label_file) 		
		
		if os.path.isfile(sys.argv[3]):
			test_file = sys.argv[3]
			image_ids, test_data = read_test_data(test_file)
		else:
			test_dir = sys.argv[3]		
			image_ids, test_data = read_test_data_dir(test_dir)

		output_file = sys.argv[4]
		
	train_data = pre_process_data(train_data)
	test_data = pre_process_data(test_data)
	temp = []
	for i in range(10):
		output = RandomForest_method(train_data, target, test_data, 100 * (i+1))
		temp.append(output)
	temp = np.array(temp)
	output = np.mean(temp,axis=0)
	#output = SVM_method(train_data, target, test_data)	
	open_file_object = open(output_file, 'wb')
	for im_id, otpt in zip(image_ids, output):
		print >> open_file_object, im_id, int(otpt)
	#return temp

if __name__ == "__main__":
	main()
