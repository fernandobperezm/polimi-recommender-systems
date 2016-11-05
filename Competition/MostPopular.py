# Politecnico di Milano
# MostPopular.py
#
# Description: This script contains the Python Code of most popular algorithm.
#
# Created by: Fernando Pérez on 25/10/2016.
#
# Last Modified: 25/10/2016.


# Importing Numpy.
import numpy as np
import operator as op
import ProcessingData as pd

# Job column is accesible by ['item_id']

# Counting the most popular jobs.
d = dict()
for job in pd.inter['item_id']:
    if (job in d):
        d[job] += 1
    else:
        d[job] = 1

# Sorting the dictionary as a list of tuples.
sorted_x = sorted(d.items(), key=op.itemgetter(1))
sorted_tuples = list(reversed(sorted_x))[0:5]

sorted_jobs = []
for (key,value) in sorted_tuples:
    sorted_jobs.append(str(key))

# Creating the string.
string = " ".join(sorted_jobs)

# Creating the array
out = np.array([])
for user in pd.tu['user_id']:
    arr = np.array([user,string])
    out = np.append(out,arr)

out = out.reshape((10000,2))
print(out[0])
print(out[0][0])

# Printing the document.
#format=["%d","%s"]
#np.savetxt("MostPopular.csv",out,fmt=format,delimiter=",",header="user_id,recommended_items")
file = open("MostPopular.csv","w")
file.write("user_id,recommended_items\n")

for row in out:
    file.write(row[0])
    file.write(",")
    file.write(row[1])
    file.write("\n")
file.close()




