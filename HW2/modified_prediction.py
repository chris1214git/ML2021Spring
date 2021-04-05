import pandas as pd
import os

# pred_path = 'prediction_ensemble_avg_record2_dnn14'
# df = pd.read_csv(os.path.join('record2', pred_path, 'prediction.csv'))
df = pd.read_csv('prediction_ensemble_avg_record2_dnn14.csv')

print(df)
df2 = df.copy()
w = []
for i in range(1, df.shape[0]-1):
	if df2.iloc[i-1, 1] == df2.iloc[i+1, 1]:
		# w.append(df2.iloc[i, 1])
		df2.iloc[i, 1] = df2.iloc[i-1, 1]

	elif df2.iloc[i, 1] != df2.iloc[i-1, 1] and df2.iloc[i, 1] != df2.iloc[i+1, 1]:
		if df2.iloc[i, 1] in [3, 8, 9]:
			df2.iloc[i, 1] = df2.iloc[i+1, 1] 

# w = pd.Series(w)
# print(w.value_counts())
# print(df2)
# df2.to_csv(os.path.join('record2', pred_path, 'prediction_modified.csv'), index=False)
df2.to_csv('prediction_ensemble_avg_record2_dnn14_modified5.csv', index=False)
