'''
#Euclidean Distance
To find the Euclidean distance between two points, we first calculate the squared distance between each dimension.
If we add up all of these squared differences and take the square root, we’ve computed the Euclidean distance.
'''

def euclidean_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        distance += (pt1[i]-pt2[i])**2
    distance = distance**0.5
    return distance

print(euclidean_distance([5, 4, 3], [1, 7, 9]))


'''
#Manhattan Distance
Manhattan Distance is extremely similar to Euclidean distance. Rather than summing the squared difference between each
dimension, we instead sum the absolute value of the difference between each dimension. It’s called Manhattan distance
because it’s similar to how you might navigate when walking city blocks.
'''

def manhattan_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        distance += abs(pt1[i]-pt2[i])
    return distance

print(manhattan_distance([1, 2], [4, 0]))


'''
#Hamming Distance
Hamming distance only cares about whether the dimensions are exactly equal. When finding the Hamming distance between two points,
add one for every dimension that has different values. Hamming distance is used in spell checking algorithms. For example,
the Hamming distance between the word “there” and the typo “thete” is one.
'''

def hamming_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt2)):
        if(pt1[i]!=pt2[i]):
            distance+=1
    return distance

print(hamming_distance([1,2], [1, 100]))



'''
#SciPy Distances
1. Euclidean Distance .euclidean()
2. Manhattan Distance .cityblock()
3. Hamming Distance .hamming()

The scipy implementation of Hamming distance will always return a number between 0 an 1, rather than summing the number of differences
in dimensions, this implementation sums those differences and then divides by the total number of dimensions.
'''

from scipy.spatial import distance

print(distance.euclidean([1,2],[4,0]))
print(distance.cityblock([1,2],[4,0]))
print(distance.hamming([5, 4, 9], [1, 7, 9]))
