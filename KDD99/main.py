import argparse
import numpy as np
import pandas as pd
from Definition import *
from sklearn.preprocessing import MinMaxScaler
import time
rseed = 93
from sklearn.externals import joblib
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='KDD99 Examples')
parser.add_argument('--dataset', type=str, default='10_percent', help='dataset to use (10_percent/full)')
parser.add_argument('--model', type=str, default='knn', help='model to process (svm/dt/knn/nb/mlp/rf/all)')
parser.add_argument('--operation', type=str, default='all', help='train/test/all')
args = parser.parse_args()
print(args, flush=True)

# ===========================================================
#Loading Data
# ===========================================================
print("Loading Data...\n", flush=True)
if args.dataset == 'full':
	data_cmb = pd.read_csv("data/kddcup.data_transformed.txt", header=None, names = col_names, low_memory=False)
else:
	data_cmb = pd.read_csv("data/kddcup.data_10_percent_transformed.txt", header=None, names = col_names, low_memory=False)
data_test = pd.read_csv("data/corrected_transformed.txt", header=None, names = col_names, low_memory=False)
X_train = data_cmb.copy().drop(['label'],axis=1)
y_train = data_cmb['label']
X_test = data_test.copy().drop(['label'],axis=1)
y_test = data_test['label']

# ===========================================================
#Dropping and Scaling 
# ===========================================================
print("Dropping and Scaling...\n", flush=True)
X_train_trans = X_train.drop(X_train.columns[[1,2,3]],axis=1)
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train_trans)
X_train_trans = scaling.transform(X_train_trans)
                                             
X_test_trans = X_test.drop(X_test.columns[[1,2,3]],axis=1)
X_test_trans = scaling.transform(X_test_trans)

# ===========================================================
#Start Training & Model Saving
# ===========================================================
if args.operation == 'train' or args.operation == 'all':
	print("Start training the",args.model, "model...", flush=True)
	time_start = time.time()
	if args.model == 'svm':
		model1 = svm.SVC(kernel='linear', C=1,verbose=True,random_state=rseed,decision_function_shape="ovo").fit(X_train_trans, y_train)
	elif args.model == 'dt':
		model1 = tree.DecisionTreeClassifier(random_state=0).fit(X_train_trans, y_train)
	elif args.model == 'knn':
		model1 = KNeighborsClassifier(n_neighbors=1).fit(X_train_trans, y_train)
	elif args.model == 'nb':
		model1 = GaussianNB().fit(X_train_trans, y_train)
	elif args.model == 'mlp':
		model1 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 6), random_state=1).fit(X_train_trans, y_train)
	elif args.model == 'rf':
		model1 = RandomForestClassifier(n_estimators = 8, criterion = "entropy").fit(X_train_trans, y_train)
	model1_name = "Models/model_"+args.model+"_on_10_percent_dataset.sav"
	print("model completed, using time %5.2f seconds" % (time.time()-time_start), flush=True)
	print("Saving Model...\n", flush=True)
	joblib.dump(model1,model1_name)

# ===========================================================
#Model information
# ===========================================================
if args.operation == 'train' or args.operation == 'all':
	model = model1
else:
	model_path = "Models/model_"+args.model+"_on_10_percent_dataset.sav"
	model = joblib.load(model_path)
print("model information:\n%s: " % model, flush=True)
print("number of labels: %d" % (model.classes_.shape[0]), flush=True)
print("expected number of classes under one-vs-one model: %d" 
      % (model.classes_.shape[0]*(model.classes_.shape[0]-1)/2), flush=True)
if args.model == 'svm':
	print("cache size: %d" % model.cache_size, flush=True)
	print("number of decisions from the model based on \'ovo\' %d" %model.decision_function(np.arange(0,38).reshape((1,-1))).shape[1], flush=True)

# ===========================================================
#Testing
# ===========================================================
if args.operation == 'test' or args.operation == 'all':
	print("Testing...\n", flush=True)
	print("accuracy based on training: %5.4f" % model.score(X_train_trans, y_train), flush=True)  
	print("accuracy based on testing: %5.4f" % model.score(X_test_trans, y_test), flush=True)  

