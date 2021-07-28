import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
'''
print(digits.DESCR)

print(digits.data)

print(digits.target[99])

plt.gray()

plt.matshow(digits.images[99])

plt.show()
'''
model = KMeans(n_clusters = 10, random_state = 55)

model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i)
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()