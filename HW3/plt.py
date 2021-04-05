import plotly.express as px
import os 
import pandas as pd

log_path = os.path.join('record', 'tf_efficientnet_b5_0')
valid_history_df = pd.read_csv(os.path.join(log_path, 'valid_history.csv'))
fig = px.line(valid_history_df, x="epoch", y="accuracy", title='valid accuracy')
fig.write_image(os.path.join(log_path, 'valid_history.png'))
