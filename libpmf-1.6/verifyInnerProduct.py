import numpy as np
a = np.loadtxt('toy-example.model')
print ("a",a)
print ("a.shape",a.shape)

b = np.loadtxt('toy-example.model2')
print ("b",b)
print ("b.shape",b.shape)

print (np.dot(a,b.T))




