import codecademylib3_seaborn
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels =  train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 50)

print(len(training_data))
print(len(training_labels))

best, best_k = 0, 1
k_list = [i for i in range(1, 101)]
accuracies = []

for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    curr = classifier.score(validation_data, validation_labels)
    accuracies.append(curr)
    if(curr>=best):
        best = curr
        best_k = k

#print(best, best_k)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
