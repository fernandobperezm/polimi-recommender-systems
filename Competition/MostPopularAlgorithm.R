# Politecnico di Milano
# MostPopularAlgorithm.r
#
# Description: This script contains the R Code for the most popular Algorithm
#              Recommendation.
#
# Created by: Fernando Pérez on 25/10/2016.
#
# Last Modified: 25/10/2016.

# The interactions (users who clicked the job i) are located in the second column
# of the data frame.
jobs <- as.data.frame(table(inter[2]))

# Getting indices by ordering by frequencies, this way we can get the 5-most jobs.
ordered <- as.data.frame(order(jobs[2],decreasing=T), col.names = "jobs")

# Getting the 5 most jobs by indexing on the jobs column.
# mostJobs <- as.data.frame(jobs[[1]])

# Writing the CSV.
#toWrite <- c(tu,mostJobs[1:5])
