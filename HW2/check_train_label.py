import pandas as pd
import os

# # pred_path = 'prediction_ensemble_avg_record2_dnn14'
# # df = pd.read_csv(os.path.join('record2', pred_path, 'prediction.csv'))

# df = pd.read_csv('train_label.csv')

# print(df)
# df2 = df.copy()
# weird_l = []

# for i in range(1, df.shape[0]-1):
# 	if df2.iloc[i-1, 1] != df2.iloc[i+1, 1] and df2.iloc[i, 1] != df2.iloc[i-1, 1]:
# 		# df2.iloc[i, 1] = df2.iloc[i+1, 1] 
# 		weird_l.append(list(df2.iloc[i-1:i+2, 1].values))
# 		print(list(df2.iloc[i-1:i+2, 1].values))

# print(weird_l)

# weird_l = pd.DataFrame(weird_l)
# weird_l.to_csv('weird_l.csv',index=False)

df = pd.read_csv('weird_l.csv')
print(df.iloc[:,1].value_counts())
print(df.iloc[:,1].nunique())