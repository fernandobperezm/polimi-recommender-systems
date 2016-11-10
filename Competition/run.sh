# Politecnico di Milano.
# run.sh: bash script for running recsys python scripts.
#!/bin/bash

~/spark-2.0.1-bin-hadoop2.7/bin/spark-submit\
  --py-files ContentBasedRecommender.py\
  My_BasicRecommender.py up.csv,ip.csv,inter.csv,tu.csv 0 ! jobroles,id,item_id

#python3 My_BasicRecommender.py up.csv,ip.csv,inter.csv,tu.csv 0 ! jobroles,id,item_id
#python3 My_BasicRecommender.py up.csv,ip.csv,inter.csv,tu.csv --header 0 --sep ! --item_key jobroles,id,item_id
#python3 basic_recommender.py ./exampledata/data.csv --header 0
