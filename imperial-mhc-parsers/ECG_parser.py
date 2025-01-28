import pandas as pd 
import altair as alt
from scipy.signal import savgol_filter
df = pd.read_json('/Users/juandelgado/Desktop/ECG-Mock-Sample.json')
a = 0
dfclean = pd.DataFrame()
for ri,row in df.iterrows():
    try:
        value = []
        for s in row['sample'][0]['data']:
            value.append(s['value']['value'])
        aux = pd.DataFrame(value,columns=['value'])
        aux['date'] = row['sample'][0]['date']
        dfclean = pd.concat([dfclean,aux],axis=0,ignore_index=True)
    except Exception as e:
        print(e)
dfclean['value_smooth'] = savgol_filter(dfclean['value'].values, 50, 3)

dfclean = dfclean.reset_index()
alt.Chart(dfclean.loc[400:550,:]).mark_line().encode(x='index',y='value_smooth').properties(width=200).save('aa6.html')
alt.Chart(dfclean).mark_line().encode(x='index',y='value_smooth').properties(width=2000).save('aa62.html')
# dfclean.to_csv('data/ECG_sample.csv',index=False)