import pandas as pd 
import numpy as np


#Read csv files
sub_fe = pd.read_csv("../Public_subs/sub_fe.csv")
tfidf_results = pd.read_csv("../Public_subs/tfidf_results.csv")
sub_xgb = pd.read_csv("../Public_subs/sub_v3.csv")

# Ensemble and create submission 

sub = pd.DataFrame()
sub['id'] = sub_fe['id']

# sub["EAP"] = 0.5*sub_fe["EAP"] + 0.5*tfidf_results["EAP"]
# sub["HPL"] = 0.5*sub_fe["HPL"] + 0.5*tfidf_results["HPL"]
# sub["MWS"] = 0.5*sub_fe["MWS"] + 0.5*tfidf_results["MWS"]

sub["EAP"] = np.exp(np.mean( [
	sub_fe["EAP"].apply(lambda x: np.log(x)), 
	tfidf_results["EAP"].apply(lambda x: np.log(x)), 
	sub_xgb["EAP"].apply(lambda x: np.log(x))
	], axis=0 ))

sub["HPL"] = np.exp(np.mean( [
	sub_fe["HPL"].apply(lambda x: np.log(x)), 
	tfidf_results["HPL"].apply(lambda x: np.log(x)), 
	sub_xgb["HPL"].apply(lambda x: np.log(x))
	], axis=0 ))

sub["MWS"] = np.exp(np.mean( [
	sub_fe["MWS"].apply(lambda x: np.log(x)), 
	tfidf_results["MWS"].apply(lambda x: np.log(x)), 
	sub_xgb["MWS"].apply(lambda x: np.log(x))
	], axis=0 ))

# sub['target'] = ( 0.3*np.exp(np.mean(
# 	[	
# 	stacked_1['target'].apply(lambda x: np.log(x)),\
# 	xgb_submit['target'].apply(lambda x: np.log(x)),\
# 	Froza_and_Pascal['target'].apply(lambda x: np.log(x)),\
# 	median_rank_submission['target'].apply(lambda x: np.log(x))\
# 	], axis =0))
# 	+ 0.3* kaggle_287['target']
# 	+ 0.15*xgb1['target']
# 	+ 0.15*xgb2['target']
# 	+ 0.1*xgb3['target']
# )


# train_df = pd.read_csv("../Data/train.csv", na_values="-1")
# y = train_df['target']
# final_gini = eval_gini(y, sub['target'])

	
sub.to_csv('../Submissions/blend_xgb_v3.csv', index=False) 