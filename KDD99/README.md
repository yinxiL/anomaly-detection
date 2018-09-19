# KDD99 Computer Network Intrusion Detection

## Dataset
DARPA collected 9 weeks of TCPdump network connectivity and system audit data, simulating various user types, various network traffic and attack methods, with the 7-week's training data probably contains more than 5,000,000 network connection records, and the remaining 2 weeks of test data probably contains 2,000,000 network connection records.

Each network connection is marked as normal or abnormal. The exception type is subdivided into 4 types of 39 types of attacks. 22 types of attacks appear in the training set, and 17 unknown types of attacks appear in the test set. Besides, the test data is not from the same probability distribution as the training data.

The four types of anomalies are:
- DOS: denial-of-service
- R2L: unauthorized access from a remote machine to a local machine
- U2R: unauthorized access to local superuser privileges by a local unpivileged user
- PROBING: surveillance and probing

The whole DataSet can be found [here](http://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data)

## Todo List
- ~Simple data pre-processing~
- ~Model selection~ 
- Training the 10 percent KDD DataSet
- Evaluation
- Improvements
- Use MapReduce or Spark to train the whole dataset
- Generalize method

## Data pre-processing
### Original
<img src="img/catagories.png" width="50%" height="50%">

Source comes from [Analyze KDD99 data set by Sean Han](https://www.youtube.com/watch?v=mm38R3NsHso)
### Steps
- Do the exploratory data analysis using `Pandas`
- Identify the target category by number
- Feature scaling
- Generate a new file in the data directory
- Import the training machine and test set in the data directory into the MongoDB database

## Models
Refer to the review of KDD99 dataset usage [1], these are the methods that are commonly used by researchers in proposed methods and for comparision.

<img src="img/Methods.png" width="40%"> <img src="img/Comparision.png" width="40%">

Below are the Classifiers that I chose for comparision.
### SVM 
This should be the best performing method according to [the results of the KDD'99 Classifier Learning Contest](http://cseweb.ucsd.edu/~elkan/clresults.html)
### Decision Tree (CART)
I build this model and the MLP one by reference to [PENGZhaoqing's work](https://github.com/PENGZhaoqing/kdd99-scikit).
### k-nearest neighbors
(Simple but good result)
### Naive Bayes 
(Bad Accuracy)
### Neural Networks (MLP)
### Random Forest

## Evaluation
The review [1] shows the usage of perform metrics
### Indicators

### Results

### Reason Analysis

## Improvements
### Use NSL DataSet [2]
- Remove duplicated data
- Remove the easiest data
- Reduce bias on normal and dos attack

### Reduce data demension
- PCA (Not perform well on this DataSet!)
- Feature selection : Use Information Coefficient to judge the degree of association (21 out of 42)
- K-means : model extraction (find each specific attack a best model)

### Avoid overfitting
- Cross validation : Only 21% of the studies from 2010 to 2015 applied cross validation [1]
- Normalization : Euclidean distance

## Usage
### Prerequisite

## Reference
[1] Özgür, Atilla and Hamit Erdem. “A review of KDD99 dataset usage in intrusion detection and machine learning between 2010 and 2015.” PeerJ PrePrints 4 (2016): e1954.

[2] M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

[3] K. Ibrahimi and M. Ouaddane, "Management of intrusion detection systems based-KDD99: Analysis with LDA and PCA," 2017 International Conference on Wireless Networks and Mobile Communications (WINCOM), Rabat, 2017, pp. 1-6


