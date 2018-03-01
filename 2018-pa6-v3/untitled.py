import numpy as np


unknown_country = np.array([-0.38, -0.17, 1.93, 0.93, 0.26, -0.25, 0.74, 0.88, -0.14, -0.21])

# countries
Vietnam = np.array([-0.88, 1.5, 1.03, -1.08, -0.58, 0.19, 0.61, -0.86, -0.24, -0.07])
England = np.array([-0.02, -1.62, 0.4, -0.63, -0.49, 1.47, -0.64, 0.16, -0.43, -0.13])
Germany = np.array([-0.21, -0.96, 0.14, -1.13, -0.1, -1.05, -0.31, 0.03, 0.71, -0.29])

# capitals
Berlin = np.array([-0.31, -0.96, -0.98, -0.17, -1.05, -1.34, 0.07, 0.04, -0.63, 0.2]) # capital of Germany
Accra = np.array([2.75, 0.86, -0.91, 0.57, 0.1, -0.07, -0.01, -0.12, -0.2, -0.67]) # capital of Ghana
Tokyo = np.array([-1.18, 0.19, -1.59, 0.52, 1.44, 0.08, 0.15, 0.01, -0.26, 0.31]) # capital of Japan
Hanoi = np.array([-0.76, 2.22, -0.69, -0.2, -0.88, 0.31, -0.53, 0.76, 0.25, 0.13]) # capital of Vietnam
Lima = np.array([-0.48, -0.05, 0.93, 2.1, -0.27, -0.19, -0.75, -0.65, 0.19, 0.14]) # capital of Peru
Tehran = np.array([2.66, 1.33, 0.01, -0.04, 0.63, -0.03, 0.07, -0.16, 0.11, -0.27]) # capital of Iran


cap_minus_country = ((Hanoi - Vietnam) + (Berlin - Germany))/2

def cosine(v1, v2):
	return v1.dot(v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))


unknown_country = unknown_country + cap_minus_country

print "Ghana %f" % cosine(unknown_country, Accra)
print "Japan %f" % cosine(unknown_country, Tokyo)
print "Peru %f" % cosine(unknown_country, Lima)
print "Iran %f" % cosine(unknown_country, Tehran)
