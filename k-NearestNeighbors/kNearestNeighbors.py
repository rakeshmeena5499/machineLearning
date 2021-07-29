from movies import movie_dataset, movie_labels

def distance(movie1, movie2):
    dist = 0
    for i in range(len(movie1)):
        dist += (movie1[i]-movie2[i])** 2
    dist = dist ** 0.5
    return dist


def classify(unknown, dataset, labels, k):
    distances = []
    #Looping through all points in the dataset
    for title in dataset:
        movie = dataset[title]
        distance_to_point = distance(movie, unknown)
        #Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])
    distances.sort()
    #Taking only the k closest points
    neighbors = distances[0:k]
    num_good = 0
    num_bad = 0
    for movie in neighbors:
        title = movie[1]
        if(labels[title]==1):
            num_good+=1
        else:
            num_bad+=1
    return 1 if num_good > num_bad else 0

print(classify([0.4, 0.2, 0.9], movie_dataset, 5))


def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
    num_correct = 0.0
    for movie in validation_set:
        guess = classify(validation_set[movie], training_set, training_labels, k)
        if(validation_labels[movie]==guess):
            num_correct+=1
    return num_correct/len(validation_set)

print(find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, 4))




#Rather than writing our own classifier every time, we can use Python’s sklearn library.

from movies import movie_dataset, labels
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(movie_dataset, labels)

print(classifier.predict([[.45, .2, .5], [.25, .8, .9],[.1, .1, .9]]))
