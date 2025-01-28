"""
ETL for PH - apple watch 
Developed by Juan Delgado, PhD - Jan 2024

This script is ...
The data is from ...

Usage: main.py plot_flag
"""
import os
import pandas as pd 
import pandas_gbq as pdg
from utils.constants import *
from utils.utils import ETL,Raw_ETL
import sys
sys.path.append('/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/imperial-mhc-parsers')
from utils.utils_imputation import ImputationWatchPhone, fit_every_patient,choose_model,impute_merge_data
# import re
# import json
imputation_flag = False
suffix = '_us2'#'_hour' '' '_nodup'

# read credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../creds/google_credentials_mhc.json"

# load the ID mapping
# dfm0 = pd.read_csv("data/clinical/MHC_IDs.csv")
dfm = pdg.read_gbq("MHC_PH.id_mapping2",project_id=project_id)

# read the data
df = pdg.read_gbq(f"""select a.* from MHC_PH.raw_activity{suffix} a
                      union all 
                      select s.*,Null duration from MHC_PH.raw_sleep{suffix} s""",project_id=project_id)

# manual correction!
df.loc[(df['patient'] == "SPVDU20050").values,'patient'] = "SPVDU20005"
df = df.query('patient!="04ee62df-475b-477b-ad74-5a206d8a1946"').reset_index(drop=True)

###### TRANSFORM THE RAW DATA MINIMALLY
rawelt = Raw_ETL(df,dfm,suffix=suffix,sink=True)
rawelt.run_clean_pipeline()
df = rawelt.df

###### CLEAN THE DATA
etl = ETL(df,suffix=suffix,sink=True)
etl.run_clean_pipeline()

if imputation_flag:
    df_clean = etl.df_clean
  # target_variable = 'StepCount'
    for target_variable in ['StepCount','FlightsClimbed',
                            'StepCountPaceMax','StepCountPaceMean',
                            'FlightsClimbedPaceMax','FlightsClimbedPaceMean']:
        print(target_variable)
      # # we are just using real data for now
      # ###### run imputation
      # # run whole imputation
      # # impute the data - correct Watch - Phone
        # df = pdg.read_gbq(f"""select * except(imputation) from MHC_PH.activity{suffix} 
        #                   where clean_status = 'clean_1'
        #                     and device in ("Watch","iPhone")
        #                   and variable = '{target_variable}'
        #                   """,project_id=project_id)
        df = df_clean.query(f"variable == '{target_variable}' and device in ('Watch','iPhone')").drop(columns=['imputation'])
        # df.to_csv('data/clean1.csv',index=False)
        # df = pd.read_csv('bkps/imputationdf.csv')
        imp = ImputationWatchPhone(df,target_variable=target_variable)
        
        # train pooled data
        imp.run_data_pipeline()
        imp.run_train_pipeline()

        # perform a fit for every curve directly - this is to characterise the curves
        Results = fit_every_patient(df,target_variable=target_variable)
        
        # choose model  
        Results = choose_model(imp,Results,target_variable,save=True)

        # impute data
        df_new = impute_merge_data(df,imp,Results)

        # sink data
        pdg.to_gbq(df_new,
                  f"MHC_PH.activity{suffix}",
                  project_id=project_id,
                  if_exists="append")
