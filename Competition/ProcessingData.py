# Politecnico di Milano
# ProcessingData.py
#
# Description: This script contains the Python Code of the processing data.
#
# Created by: Fernando Pérez on 17/10/2016.
#
# Last Modified: 25/10/2016.


# Importing Numpy.
import numpy as np

# Pre-Processing Data.
# R CODE:

# user_profile <- read.delim("~/Development/usb-projects/polimi-projects/Recommender Systems/Competition/user_profile.csv",na.strings = c("","NA","NULL"))
# write.table(user_profile,"up.csv", na=0, sep="!",quote=F, row.names=F)

# Processing Data.
# Interaction has all columns filled.
inter = np.genfromtxt("interactions.csv", names=True, dtype=int)

# Item Profile doesn't have all columns filled.
#   * Title: List. Position in Array -> 1
#   * Country: String. Position in Array -> 5
#   * Tags: List. Position in Array -> 10
#item_profile = np.genfromtxt("item_profile.csv", names=True,\
#                             missing_values={None:"NULL"}, filling_values={2:b"0 ",0:0,1:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0},\
#                             dtype = None)

# Target Users.
tu = np.genfromtxt("target_users.csv", names=True, dtype=int)

# Each user profile
#user_profile = np.genfromtxt("up.csv",delimiter="!", names=True, dtype = None)


#print(interactions)
#print(item_profile)
#print(target_users)
#print(user_profile)


