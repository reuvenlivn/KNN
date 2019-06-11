import numpy as np
from sklearn import datasets
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt

iris_path=''

class DataSet():
    def __init__(self, X,Y):
        assert len(X)==len(Y)   # sanity check
        self.X=X
        self.Y=Y
        
    """ 
    returns X_train, Y_train, X_test, Y_test 
        train_set  = p_train * all data
    """
    def make_train_test(self,p_train):
        assert 0<p_train<1      # sanity check
        data_len=len(self.X)
        train_len=round(p_train*data_len)
        indexes = np.random.permutation(data_len)
#       print('perm',perm)
        train_indexes = indexes[:train_len]
#       print('train_pos',train_pos)      
        test_indexes = indexes[train_len:]
#       print('test_pos',test_pos)      
        return (self.X[train_indexes],self.Y[train_indexes],self.X[test_indexes],self.Y[test_indexes])
    
    # property - enables calling function as as an attribute (this is a get property, 
    # set property also exist but has a different synthax)
    @property
    def std(self):
        return (self.X.std(axis=0),self.Y.std(axis=0))

    @property
    def mean(self):
        return (self.X.mean(axis=0),self.Y.mean(axis=0))
        
# deistance between x and y
def distance(x,y):
    return (np.linalg.norm(x-y))

    
class KNN():
    def __init__(self, X_train, Y_train, k):
        self.X_train=X_train
        self.Y_train=Y_train
        self.k=k

    @staticmethod  #static method - a method which does not depend on an 
    # instance and acts exactly like any standalone method outside of a classe
    # hence self is not passed as an argumetn
    def majority_vote(y_labels):
        return(np.bincount(y_labels).argmax())

    # find the neibours of a given point       
    def _k_neighbours(self,point):
        distances=[]
        # loop over all the train set, claculate the distances from point to 
        # all the train set
        for i in range(len(self.X_train)):
            distances.append((self.Y_train[i],
                              distance(point, self.X_train[i]))
                            )
        # sort the distances accoring to the second value in the tuple
        distances.sort(key=lambda tup: tup[1])
        # return only the K nearest points
        return distances[:self.k]
    
    def _classify_point(self,point):
        k_neighbours=self._k_neighbours(point)
        labels=[tup[0] for tup in k_neighbours]
        # we call the static method using the class name (though self also works)
        return KNN.majority_vote(labels) 
        
    def test(self,X_test):
        results=[]
        for i in range(len(X_test)):
            results.append(self._classify_point(X_test[i]))
        return(np.array(results))

# ckaculate success in percentage
def calc_success(Y_test, Y_classified):
    assert len(Y_test) == len(Y_classified)
    return ((Y_test==Y_classified).sum() / len(Y_test))

## main
iris = datasets.load_iris()

data_set = DataSet(iris.data, iris.target)

# This is not part of KNN, just class demo
print('Data set mean: {}, and standart deviation: {}'.format(data_set.mean, data_set.std))

X_train,Y_train,X_test,Y_test = data_set.make_train_test(0.75)
knn = KNN(X_train, Y_train, 7)
Y_classified = knn.test(X_test)
print('Success percentage {:.2%}'.format(calc_success(Y_test, Y_classified)))

iris_feature_names = pd.DataFrame(iris.feature_names)
iris_target_names = pd.DataFrame(iris.target_names)

iris_sns = sns.load_dataset("iris")
sns.pairplot(iris_sns, hue="species", palette="dark",
             height=4, markers=["p", "s", "X"]);

