import numpy as np
a = np.loadtxt('toy-example.model.W')
#print ("a",a)
print ("a.shape",a.shape)

b = np.loadtxt('toy-example.model.H')
#print ("b",b)
print ("b.shape",b.shape)

print (np.dot(a,b.T))




