import pandas as pd
import pandas_gbq as pdg
from utils.constants import *
from utils.mhctools import SynapseCacheZip
import os

suffix = '_nodup'
Ext_ID = 'SPVDU00165'
if_exists = 'append'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../creds/google_credentials_mhc.json"


d2 = pdg.read_gbq(f'select * from MHC_PH.raw_activity{suffix} limit 1', project_id=project_id)
keys = d2.columns

ext_ids = pdg.read_gbq(f"""select distinct patient from MHC_PH.raw_activity 
                       where patient not in (select distinct patient from MHC_PH.raw_activity{suffix})
                       """, project_id=project_id)

for Ext_ID in ext_ids.values:
    Ext_ID = Ext_ID[0]
    d = pd.read_csv(f'/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/imperial-mhc-parsers/bkps/{Ext_ID}bkp.csv')
    hk = d.loc[:,keys]

    sc = SynapseCacheZip(pathname,zipname)

    # save the raw by hourly data
    hk_hour = sc.aggregate_hkdata(hk,date_format='%Y-%m-%d-%H')
    pdg.to_gbq(hk_hour, f"MHC_PH.raw_activity_hour{suffix}", project_id=project_id, if_exists=if_exists)
    print('activity (hourly) uploaded to GBQ')

    # aggregate the data
    hk_out = sc.aggregate_hkdata(hk,date_format='%Y-%m-%d')
    # save
    pdg.to_gbq(hk_out, f"MHC_PH.raw_activity{suffix}", project_id=project_id, if_exists=if_exists)
    print('activity uploaded to GBQ')


