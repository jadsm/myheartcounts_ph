from utils.mhctools import SynapseCacheZip,complete_loop
from utils.utils import delete_tables
import pandas as pd
import numpy as np
import os
from utils.constants import *
import pandas_gbq as pdg
from datetime import datetime
from multiprocessing import Pool
import logging

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../creds/google_credentials_mhc.json"

parallel = False
include_all_other_patients = True # not only the ones with ids matching ourcohort
n_pools = 6
full_reload = False # this flags controls whether you will load a delta or completely
tables_to_load = ['workout']#'6mwt','workout','sleep','activity'  
suffix = '_us2'

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename=f"logs/info_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log", level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)

sc = SynapseCacheZip(pathname,zipname,include_all_other_patients=include_all_other_patients)

# Here we just print all the External IDs that we are going to used for extraction
list_Ext_IDs_All = np.unique(sc.id_link.external_identifier) 

if not full_reload:
    try: 
        list_Ext_IDs_in_DB = pdg.read_gbq(f'select patient from MHC_PH.raw_{tables_to_load[0]}{suffix} group by patient order by count(*) asc', project_id=project_id).values.reshape(-1)
    except:
        list_Ext_IDs_in_DB = []

    # list_Ext_IDs_in_DB = pdg.read_gbq('select distinct patient from MHC_PH.raw_activity', project_id=project_id).values.reshape(-1)
    print('IDs in DB:',len(list_Ext_IDs_in_DB))
    list_Ext_IDs_All = [k for k in list_Ext_IDs_All if k not in list_Ext_IDs_in_DB]
    print('IDs to be computed:',len(list_Ext_IDs_All))

    list_Ext_IDs_All = [l for l in list_Ext_IDs_in_DB if l in list_Ext_IDs_All] + [l for l in list_Ext_IDs_All if l not in list_Ext_IDs_in_DB]

    if len(list_Ext_IDs_All) == 0:
        print('No new data to load - please select full_reload=True to reload all data or add new data to the database.')
        exit()
    else:
        print(f"Loading data for {len(list_Ext_IDs_All)} new patients not currently in the database.")

elif full_reload:
    print(f"Re-loading all data by coercion full_reload=True.")


if __name__ == '__main__':

    if full_reload:
        answer = input(f"""Full reload mode will wipe these tables: {','.join(tables_to_load)}. 
                       Are you sure you want to continue? Y/n""")
        if answer.lower() in ["y","yes"]:
            print('wiping tables...')
            # wipe the tables
            for table in tables_to_load:
                delete_tables(table,suffix) 
                # pass    
                
        else:
            print('aborting...')
            exit()

    # list_Ext_IDs_All = ['SPVDU03004']
    fixed_attr = (logger,sc,suffix,tables_to_load)
    input_data = (tuple([Ext_ID])+fixed_attr for Ext_ID in list_Ext_IDs_All)
    
    if parallel:
        # execute the loop in parallel
        with Pool(n_pools) as p:  
            p.starmap(complete_loop, input_data)
    else:
        for params in list(input_data):
            complete_loop(*params)
    print('FINISHED!!!')
            
