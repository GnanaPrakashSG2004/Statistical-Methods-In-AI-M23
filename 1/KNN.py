import sys                          # To get the path argument of the test data file
import numpy as np                  # For general array handling
import pandas as pd                 # For printing the metrics in a table format
from sklearn import metrics as sm   # For calculating the given metrics of the model

#############################################################################################################################################################

test_data_path = sys.argv[1]        # If the script is invoked using the command `python KNN.py test.npy`, then, the path to the test data is given in the second argument

data = np.load("data.npy", allow_pickle=True)
test_data = np.load(test_data_path, allow_pickle=True)

#############################################################################################################################################################

if data.shape[1] != test_data.shape[1]:
    raise ValueError(f'Number of features in train and test data do not match. Train data has {data.shape[1]} features and test data has {test_data.shape[1]} features')

data      = data[:, 1:4]      # Removing the guess time column
test_data = test_data[:, 1:4] # and the data IDs

for i in test_data[:, 0]:
    if i.shape[1] != 1024:
        raise ValueError(f'Number of features in ResNet encoding must be 1024. Found {i.shape[1]} features in test data')

for i in test_data[:, 1]:
    if i.shape[1] != 512:
        raise ValueError(f'Number of features in VIT encoding must be 512. Found {i.shape[1]} features in test data')

#############################################################################################################################################################

np.random.shuffle(data)

train_validation_split = 0.8    # 80-20 train validation split of the dataset
train_data_len = int(train_validation_split * data.shape[0])

train_data = data[:train_data_len]      # First 80% of the shuffled dataset
validation_data = data[train_data_len:] # Last  20% of the shuffled dataset

train_labels = np.array([i for i in train_data[:, 2]])

train_resnet_data       = np.array([i[0] for i in train_data[:, 0]])       # All the encodings are given as list of list and not just a list. So,
train_vit_data          = np.array([i[0] for i in train_data[:, 1]])       # these loops extract the inner encoding and create a new list of encodings
validation_resnet_data  = np.array([i[0] for i in validation_data[:, 0]])    
validation_vit_data     = np.array([i[0] for i in validation_data[:, 1]])

#############################################################################################################################################################

# Function to print the metrics in a table format
def printMetrics(vit_metrics_values):
    metrics_list = ['Accuracy', 'Macro F-1 score', 'Micro F-1 score', 'Weighted F-1 score', 'Precision', 'Recall']
    table_list   = []

    for i in range(len(metrics_list)):
        table_list.append([metrics_list[i], vit_metrics_values[i]])
    
    # Printing the metrics in a table format after removing the index column
    df = pd.DataFrame(table_list, columns=['Metric', 'Value'])
    blankIndex = [''] * len(df)
    df.index = blankIndex
    print(df)
    print()

#############################################################################################################################################################

# This class implements all the three distance metrics given
class DistMetric():
    def __init__(self):
        pass

    # In all these functions, axis=1 has been used in the norm function to calculate norm of each column of the train_encoding matrix
    # as each column of the matrix will be one encoding of a data point. So, the norm of each column will be the distance between the
    # train data point and the test data point
    
    def __euclidean(self, train_encoding, test_encoding) -> float:
        return np.linalg.norm(train_encoding - test_encoding, axis=1)

    def __manhattan(self, train_encoding, test_encoding) -> float:
        return np.linalg.norm(train_encoding - test_encoding, axis=1, ord=1)
    
    def __cosine(self, train_encoding, test_encoding) -> float:
        return 1 - ((train_encoding @ test_encoding) / (np.linalg.norm(train_encoding, axis=1) * np.linalg.norm(test_encoding)))
    
    def calcDist(self, train_encoding, test_encoding, metric_type='Euclidean') -> float:
        if metric_type.lower() == 'euclidean':
            return self.__euclidean(train_encoding, test_encoding)
        elif metric_type.lower() == 'manhattan':
            return self.__manhattan(train_encoding, test_encoding)
        elif metric_type.lower() == 'cosine':
            return self.__cosine(train_encoding, test_encoding)
        else:
            raise ValueError(f'{metric_type} is not a valid distance metric')
        
# Class containing the implementation of the optimised KNN model
class KNN(DistMetric):
    # Initialisation method
    def __init__(self, encoder='ResNet', k=3, dist_metric='Euclidean'):
        self.encoder              =  encoder
        self.k                    =  k
        self.dist_metric          =  dist_metric
        self.train_encodings      =  []   # List to keep track of encodings of all train data points
        self.validation_encodings =  []   # List to keep track of encodings of all validation data points
        self.test_encoding        =  []   # List to keep track of encoding of the given test data point
        self.train_labels         =  []   # List to keep track of labels of all train data points 
        
        self.__train()

    # Setting the train encodings based on the encoder type and also setting the labels
    def __train(self):
        if self.encoder.lower()   == 'resnet':
            self.train_encodings   = train_resnet_data
        elif self.encoder.lower() == 'vit':
            self.train_encodings   = train_vit_data
        self.train_labels          = train_labels

    # Method to predict the label of the given data point
    def __predict(self, data) -> str:
        dist_list = DistMetric().calcDist(self.train_encodings, data, self.dist_metric)  # List of distances between all records in the train dataset and the test data point
        
        # Next 4 lines of code have been generated by ChatGPT -> Prompts written in `1.ipynb` file
        
        dist_indices = np.argsort(dist_list)                            # Returns a list of the indices where the distances have been placed after sorting
        k_nearest_labels = self.train_labels[dist_indices][:self.k]     # Extracting the labels ordered using the above indices and then selecting the first k labels

        labels, freqs = np.unique(k_nearest_labels, return_counts=True) # Returns the unique labels and their frequencies in the k nearest labels
        return labels[np.argmax(freqs)]                                 # Returns the index in the list which has the highest frequency which is then used to find the corresponding label

    # Method to perform model validation which returns a list of the predicted labels corresponding to the validation set
    def inference(self) -> list:
        if self.encoder.lower()        == 'resnet':
            self.validation_encodings   = validation_resnet_data
        elif self.encoder.lower()      == 'vit':
            self.validation_encodings   = validation_vit_data

        validation_labels = [self.__predict(i) for i in self.validation_encodings]
        return validation_labels
    
    # Method to return the predicted label of the given data point from the test set
    def predict(self, data) -> str:
        if self.encoder.lower()   == 'resnet':
            self.test_encoding     = data[0][0]
        elif self.encoder.lower() == 'vit':
            self.test_encoding     = data[1][0]

        return self.__predict(self.test_encoding)

#############################################################################################################################################################

# Below are the parameters obtained by hyperparameter tuning done in 1.ipynb:
# Best for VIT    -> k = 6, Distance metric = Cosine
# Best for ResNet -> k = 9, Distance metric = Manhattan

vit_model    = KNN(encoder='VIT',    k=14, dist_metric='Manhattan'   )
resnet_model = KNN(encoder='ResNet', k=7 , dist_metric='Manhattan')

test_data_labels = test_data[:, 2]  # Last column of the test data contains the labels

#############################################################################################################################################################

vit_model_preds = [vit_model.predict(i) for i in test_data]

vit_accuracy    = sm.accuracy_score( test_data_labels, vit_model_preds)
vit_macro_f1    = sm.f1_score(       test_data_labels, vit_model_preds, average='macro')
vit_micro_f1    = sm.f1_score(       test_data_labels, vit_model_preds, average='micro')
vit_weighted_f1 = sm.f1_score(       test_data_labels, vit_model_preds, average='weighted')
vit_precision   = sm.precision_score(test_data_labels, vit_model_preds, average='macro', zero_division=0)
vit_recall      = sm.recall_score(   test_data_labels, vit_model_preds, average='macro', zero_division=0)

vit_metrics = [vit_accuracy, vit_macro_f1, vit_micro_f1, vit_weighted_f1, vit_precision, vit_recall]
print('VIT metrics:')
printMetrics(vit_metrics)

#############################################################################################################################################################

resnet_model_preds = [resnet_model.predict(i) for i in test_data]

resnet_accuracy    = sm.accuracy_score( test_data_labels, resnet_model_preds)
resnet_macro_f1    = sm.f1_score(       test_data_labels, resnet_model_preds, average='macro')
resnet_micro_f1    = sm.f1_score(       test_data_labels, resnet_model_preds, average='micro')
resnet_weighted_f1 = sm.f1_score(       test_data_labels, resnet_model_preds, average='weighted')
resnet_precision   = sm.precision_score(test_data_labels, resnet_model_preds, average='macro', zero_division=0)
resnet_recall      = sm.recall_score(   test_data_labels, resnet_model_preds, average='macro', zero_division=0)

resnet_metrics = [resnet_accuracy, resnet_macro_f1, resnet_micro_f1, resnet_weighted_f1, resnet_precision, resnet_recall]
print('ResNet metrics:')
printMetrics(resnet_metrics)

#############################################################################################################################################################