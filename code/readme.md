
##################### keep 25% of training data ###################

Because we only want to train a small subset of the data, we want to keep 25% of the data. What it does it to remove the 75 percent of the users. For the remaining 25% of the users, it is still going to be to use the last interaction for recommendation. 
(1) bash filter_user.sh




#################### generate relevant scores to filter out better pretraining data ################
(1) bash gen_relevance_score_amazon.sh 

for pixelrec result in the excel sheet, the data before generating the scores is "epoch_6_clueweb300k_train.csv" in the datasets/ and "information/" folder. 

