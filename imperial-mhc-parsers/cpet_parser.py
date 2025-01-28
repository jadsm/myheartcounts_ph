import pandas as pd
import pandas_gbq as pdg
import os
google_credentials_path = "../../creds/google_credentials_mhc.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path 
paths = ['/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/clinical/CPET-PH[4496].csv', 
         '/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/clinical/CPET-Cube[4495].csv']

df_cpetPH = pd.read_csv(paths[0])
df_cpetCube = pd.read_csv(paths[1])

# aa = df_cpetPH.merge(df_cpetCube,on=['SPVDU  ID','Test Date'],how='outer',indicator=True)
df_cpetPH.columns = [c.replace(" ",'').replace("-",'').replace(r"/",'_') for c in df_cpetPH.columns]


df_cpetPHA = df_cpetPH.melt(id_vars=['DateofConsent', 'SPVDUID', 'ExerciseTestDate'],value_vars=['WalkingDistance','Presats', 'Postsats']).rename(columns={'ExerciseTestDate':'TestDate'})
df_cpetPHB = df_cpetPH.melt(id_vars=['DateofConsent', 'SPVDUID','TestDate'],value_vars=df_cpetPH.keys()[8:])
df_cpetPH_out = pd.concat([df_cpetPHA,df_cpetPHB],axis=0,ignore_index=True)

df_cpetPH_out = df_cpetPH_out.dropna(subset=['value']).reset_index(drop=True)

# transform BP 
aux = df_cpetPH_out.query("variable in ('BP_Rest', 'BP_Exercise')")
import re

pattern = r'(\d+)(?:\.|\/)(\d+)?'

aux['high'] = aux['value'].apply(lambda x: re.search(pattern, x).group(1))
aux['low'] = aux['value'].apply(lambda x: re.search(pattern, x).group(2))
aux = aux.drop(columns=['value']).melt(id_vars=['DateofConsent', 'SPVDUID','TestDate','variable'],var_name='level')
aux['variable'] += aux['level']
aux.drop(columns=['level'],inplace=True)

df_cpetPH_out = pd.concat([df_cpetPH_out.query("variable not in ('BP_Rest', 'BP_Exercise')"),
                            aux],axis=0,ignore_index=True)

df_cpetPH_out = df_cpetPH_out.dropna(subset=['value']).reset_index(drop=True)
df_cpetPH_out['value'] = df_cpetPH_out['value'].astype(float)
pdg.to_gbq(df_cpetPH_out.dropna(subset=['TestDate']),'MHC_PH.CPETPH',project_id='imperial-410612',if_exists='replace')
