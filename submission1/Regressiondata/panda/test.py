import numpy as np

a = np.arange(1,10).reshape(3,3)
print(a)
b = np.arange(10,19).reshape(3,3)
print(b)


print(a*b)

c = np.array([[1,2],[3,4],[5,6]])
print(c.shape)