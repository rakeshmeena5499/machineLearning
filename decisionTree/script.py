'''
#Gini Impurity
This idea can be quantified by calculating the Gini impurity of a set of data points. To find the Gini impurity,
start at 1 and subtract the squared percentage of each label in the set. 
'''

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity



'''
#Information Gain
Information gain measures difference in the impurity of the data before and after the split. For example, let’s say you
had a dataset with an impurity of 0.5. After splitting the data based on a feature, you end up with three groups with 
impurities 0, 0.375, and 0. The information gain of splitting the data in that way is 

			0.5 - 0 - 0.375 - 0 = 0.125

Let’s modify the formula for information gain to reflect the fact that the size of the set is relevant. Instead of simply 
subtracting the impurity of each set, we’ll subtract the weighted impurity of each of the split sets. If the data before 
the split contained 20 items and one of the resulting splits contained 2 items, then the weighted impurity of that subset 
would be 2/20 * impurity. We’re lowering the importance of the impurity of sets with few elements.
'''

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= len(subset)/len(starting_labels)*gini(subset)
  return info_gain


'''
#Build Tree
Now that we can find the best feature to split the dataset, we can repeat this process again and again to create the full tree. 
This is a recursive algorithm! We start with every data point from the training set, find the best feature to split the data, 
split the data based on that feature, and then recursively repeat the process again on each subset that was created from the split.
'''

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets


def build_tree(data, labels):
  best_feature, best_gain = find_best_split(data, labels)
  if(best_gain==0):
    return Counter(labels)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branches.append(build_tree(data_subsets[i], label_subsets[i]))
  return branches

def classify(datapoint, tree):
  if(isinstance(tree, Leaf)):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]
  value = datapoint[tree.feature]
  for branch in tree.branches:
    if(branch.value == value):
      return classify(datapoint, branch)  


'''
#Decision Trees in scikit-learn
The sklearn.tree module contains the DecisionTreeClassifier class.
'''

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth = 12)
classifier.fit(training_points, training_labels)
print(classifier.score(testing_points, testing_labels))
print(classifier.tree_.max_depth)
