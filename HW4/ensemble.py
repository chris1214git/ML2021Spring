import pandas as pd
import os
import numpy as np
from scipy import stats

select_folders = ['output_HW04-conform-sa16_dmodel300.csv', 'output_HW04-conform-sa16_dmodel200_dropout0.2_l10.000001.csv', \
'output_HW04-conform-sa16_dmodel160.csv']

preds = []
for folder in select_folders:
	print(folder)
	df = pd.read_csv(folder)
	preds.append(df.values[:,1])

preds = np.array(preds)
preds = stats.mode(preds)
print(preds[0].shape)

df.iloc[:,1] = preds[0].flatten()
print(df)
df.to_csv('ensemble_1.csv', index=False)


# ensemble 1
# select_folders = ['tf_efficientnet_b5_lr4_scheduler2_dropout2', 'tf_efficientnet_b5_lr4_scheduler2_dropout3',\
#  'tf_efficientnet_b5_lr4_scheduler2_dropout3-2-epoch160',\
# 'tf_efficientnet_b5_lr4_scheduler2_dropout3_randaug', 'tf_efficientnet_b5_lr4_scheduler2_dropout3_randaug2', \
# 'efficientnet_b5_opti_2', 'tf_efficientnet_b5_lr4_scheduler2_dropout3_semi_balance_1'  ]
