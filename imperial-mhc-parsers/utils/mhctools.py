import re
import json
import pandas as pd
import pandas_gbq as pdg
import numpy as np
import os
from pathlib import Path
from io import StringIO
from tqdm.auto import tqdm
from typing import Union, Any
import subprocess
import zipfile
from datetime import datetime
from utils.constants import *


class SynapseCache:
    """Read & parse entries from a synapseCache directory

    Raises:
        FileNotFoundError: If the directory didn't exist
    """

    def __init__(self, base_path: Union[str, os.PathLike],zipname: str):
        """Construct a new SynapseCache object

        Args:
            base_path (Union[str, os.PathLike]): the root folder where the synapseCache exists.
        """
        self.base_path = Path(base_path)
        self.cache_path = self.base_path / 'synapseCache'
        self.hkcache = self.base_path / 'hkcache'
        
        self.tables = {}
        
        # Look for tableQueries.tab in a couple of possible locations
#         tablequeries_locations = [
#             self.base_path / "tableQueries.tab",
#             self.cache_path / "tableQueries.tab"
#         ]
#         for path in tablequeries_locations:
#             if path.is_file():       
#                 with open(path) as tsv:
#                     for table_name, file_path in csv.reader(tsv, dialect="excel-tab"):
#                         self.tables[table_name] = Path(file_path).parent.name
#                 break;
        
#         if not self.tables:
#             raise FileNotFoundError(f"Coudn't find tableQueries.tab")

        # Load external identifiers
        ext_ids = pd.read_csv(self.table_path('ExternalIdentifier-v1'),low_memory=False)
        
        
        # Make the ext. ID field uppercase
        ext_ids.external_identifier = ext_ids.external_identifier.str.upper()

        # Create link table
        self.id_link = ext_ids[['healthCode', 'external_identifier']].drop_duplicates().set_index('healthCode')

    def table_path(self, table_name: str) -> Path:
        """Returns the full filesystem path of the file containing the requested table.
        
        Will look first in the 'table' subfolder of synapseCache, in case the table has already been moved there as part of data preparation.

        Args:
            table_name (str): The name of the table required

        Returns:
            Path: The path to the table's file
        """
        
        table_cache_file = (self.cache_path / 'table' / table_name).with_suffix('.csv')
        #print(table_cache_file)  #add Juan
        #print(table_cache_file.is_file()) #add Juan

        if table_cache_file.is_file():
            return table_cache_file
        
        return self.blob_path(self.tables[table_name]) 

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a table by name, adding an external_identifier column

        Args:
            table_name (str): The name of the table required

        Returns:
            pd.DataFrame: The table data
        """

        # Get the list of columns in this table
        cols = pd.read_csv(self.table_path(table_name), nrows=0).columns

        # We will treat everything as a string, except for a few specific columns
        dtypes = {c: str for c in cols}
        dtypes['ROW_ID'] = 'Int64'
        dtypes['ROW_VERSION'] = 'Int64'
        dtypes['dayInStudy'] = 'Int64'

        return pd.read_csv(self.table_path(table_name),
                           dtype=dtypes,
                           parse_dates=["uploadDate", "createdOn"],
                           infer_datetime_format=True).join(self.id_link, on='healthCode')

    def blob_path(self, blob_id: str) -> Path:
        """Get the path to a synapse cache blob by its ID

        This will return the path of the first non-hidden file found in the appropriate directory.

        Args:
            blob_id (str): The blob identifier

        Raises:
            FileNotFoundError: If there are no files found at the blob's directory

        Returns:
            Path: Path to the blob data
        """

        # Convert blob_id from int to str, just in case
        blob_id = str(blob_id)

        # Build the path
        path = self.cache_path / str(int(blob_id[-3:])) / blob_id
        #print("blob_id:",path)     #add Juan

        # Look for all files containing data.csv
        try:
            file_path = next(f for f in path.iterdir() if f.is_file() and not f.name.startswith('.'))
        except StopIteration:
            raise FileNotFoundError("Blob path contains no files")
        else:
            return file_path

    def load_json(self, blob_id: str) -> Any:
        """Load a JSON blob

        Args:
            blob_id (str): The blob identifier

        Returns:
            Any: The parsed JSON data as a Python object
        """
        try:
            with open(self.blob_path(blob_id), 'r') as f:
                #print("path to load json:",self.blob_path(blob_id))   #add Juan
                return json.load(f)
        except Exception as e:
            print(e)
            return 
    def load_data(self, blob_id: str) -> pd.DataFrame:
        """Loads a dataset from a data.csv file

        Args:
            blob_id (str): The blob identifier for the data.csv

        Returns:
            pd.DataFrame: Data output
        """
        try:
            # There are some items in these files which have commas in an unquoted device name field.
            # For reference, a list of all possible device IDs can be found at:
            # https://gist.githubusercontent.com/adamawolf/3048717/raw/19b4cc0627ff669f1c4589324b9cb45e4948ec01/Apple_mobile_device_types.txt

            # The following regex will match any (current) Apple device ID that contains commas:
            device_re = re.compile(r"(?:iPhone|iPad|iPad|Watch)\d+,\d+")

            # Function to surround with quotes any device IDs which contain commas
            def fix_bad_device_ids(path):
                with open(path) as f:
                    return StringIO('\n'.join(device_re.sub(r'"\g<0>"', row) for row in f))

            # Read in the data
            data = pd.read_csv(fix_bad_device_ids(self.blob_path(blob_id)),
                            dtype={'value': np.float64},
                            names=['startTime',
                                    'endTime',
                                    'type',
                                    'value',
                                    'unit',
                                    'source',
                                    'sourceIdentifier',
                                    'appVersion'],
                            parse_dates=['startTime', 'endTime'],
                            date_parser=lambda col: pd.to_datetime(col, utc=True, errors='coerce'),
                            header=None,
                            quotechar='"',
                            na_values=['value'],
                            #    error_bad_lines=False,# deprecated since version 1.3.0
                                on_bad_lines='warn',
                            #    warn_bad_lines=False,
                            index_col=False,
                            engine='c')

            # Strip out any header rows that have been picked up
            return data[data.type != 'type']
        except Exception as e:
            print(e)
            return pd.DataFrame([])

    def get_healthkit_data(self, external_id: str, no_cache: bool = False, limit: int = None) -> pd.DataFrame:
        """Returns the full healthkit dataset for the given external_id

        Args:
            external_id (str): The identifier to be filtered for
            no_cache (bool, optional): If True, will not cache the data or use a previously-saved cache. Defaults to
                                       False.
            limit (int, optional): If provided, limit only to this number of entries. Useful for testing.

        Returns:
            pd.DataFrame: The HealthKit dataset
        """
        # Make sure extid is capitalised
        external_id = external_id.upper()
    
        # Load from cache if we can
        cache_file = (self.hkcache / external_id).with_suffix('.pickle')

        if not no_cache and cache_file.is_file():
            #print("IN")   #add Juan
            return pd.read_pickle(cache_file)

        # Otherwise, load the healhkit table from the synapseCache if this hasn't been done yet
        if not hasattr(self, 'hktable'):
            # self.hktable = self.load_table("cardiovascular-HealthKitDataCollector-v1")
            self.hktable = pd.read_csv(self.zip_table_like_path("cardiovascular-HealthKitDataCollector-v1"))

        # Filter the healthkit table by external ID and grab data.csv column
        blobs = self.hktable[self.hktable.external_identifier == external_id]['data.csv']

        # Set limit if requested
        if limit:
            blobs = blobs[0:limit]

        # Concatenate all data.csv files for this ID
        data_csvs = (self.load_data(blob) for blob in blobs.values)
        # data_csvs = (self.load_data(blob) for _, blob in blobs.iteritems())

        # Concatenate the results, wrapping our generator with tqdm for a progress bar
        df = pd.concat(tqdm(data_csvs, total=len(blobs), desc="Loading healthkit data"),
                       ignore_index=True)

        # Ensure cache dir exists
        self.hkcache.mkdir(parents=True, exist_ok=True)

        # Save to cache and return
        if not no_cache:
            df.to_pickle(cache_file)
        return df

    def get_6mwt_data(self, external_id: str) -> pd.DataFrame:
        """Gets the 6-minute walk test data for the given ID

        Args:
            external_id (str): The external ID to select data for

        Returns:
            pd.DataFrame: Resulting dataset
        """
        # Load the table data
        wtdf = self.load_table("6-Minute Walk Test_SchemaV4-v6")
        #print("wtdf: ", wtdf)   #add Juan
        
        # Pull the json blob for each blob_id in the pedometer_fitness.walk column; we only want the last row in 'items'
        jsons = (self.load_json(blob_id)['items'][-1]
                 for blob_id in wtdf[wtdf.external_identifier == external_id]["pedometer_fitness.walk"].dropna())
        
        #for blob_id in wtdf[wtdf.external_identifier == external_id]["pedometer_fitness.walk"].dropna():
        #    print("blob_id in for:",blob_id)
        #    print("wtdf.external_identifier: ",wtdf.external_identifier)
        #    print("external_id: ",external_id)
            #print("wtdf.external_identifier==external_id:",wtdf.external_identifier == external_id)
            #print("wtdf:",wtdf[wtdf.external_identifier == external_id]["pedometer_fitness.walk"].dropna())
            
        # Create a DataFrame and put the columns in a consistent order
        df = pd.DataFrame(jsons, columns=['startDate', 'endDate', 'numberOfSteps', 'distance', 'floorsAscended',
                                          'floorsDescended'])

        # Turn startDate and endDate into datetime64s
        for col in ('startDate', 'endDate'):
            df[col] = pd.to_datetime(df[col], utc=True)

        # Index on startDate for convenience
        return df.set_index('startDate')

class SynapseCacheZip:
    """Read & parse entries from a synapseCache directory

    Raises:
        FileNotFoundError: If the directory didn't exist
    """

    def __init__(self, base_path: Union[str, os.PathLike],zipname: str,usecols=None,include_all_other_patients=False):
        """Construct a new SynapseCache object

        Args:
            base_path (Union[str, os.PathLike]): the root folder where the synapseCache exists.
        """
        self.base_path = Path(base_path)
        self.cache_path = self.base_path / 'synapseCache'
        self.hkcache = self.base_path / 'hkcache'
        self.zipname = zipname
        self.synapse_zip_path = 'UK_Export/synapseCache'
        self.zf = zipfile.ZipFile(f'{self.base_path}/{self.zipname}') 
        self.include_all_other_patients = include_all_other_patients
        how = 'outer' if self.include_all_other_patients else 'inner'

        self.tables = {}
        
        # Load external identifiers
        # ext_ids = pd.read_csv(self.table_path('ExternalIdentifier-v1'))
        ext_ids = pd.read_csv(self.zip_table_like_path('ExternalIdentifier-v1'))
        
        # Make the ext. ID field uppercase
        ext_ids.external_identifier = ext_ids.external_identifier.str.upper()

        # Create link table
        self.id_link = ext_ids[['healthCode', 'external_identifier']].drop_duplicates()#.set_index('healthCode')
        
        # create hk table - activity
        hktable = pd.read_csv(self.zip_table_like_path("cardiovascular-HealthKitDataCollector-v1"))
        self.hktable = hktable.merge(self.id_link, on='healthCode',how=how)
        # create hk table - sleep
        sleeptable = pd.read_csv(self.zip_table_like_path("cardiovascular-HealthKitSleepCollector-v1"))
        self.sleeptable = sleeptable.merge(self.id_link, on='healthCode',how=how)
        # create hk table - workout
        workouttable = pd.read_csv(self.zip_table_like_path("cardiovascular-HealthKitWorkoutCollector-v1"))
        self.workouttable = workouttable.merge(self.id_link, on='healthCode',how=how)
        # Load the table data
        wtdf = pd.read_csv(self.zip_table_like_path("'6-Minute Walk Test_SchemaV4-v6'"))
        self.wtdf = wtdf.merge(self.id_link, on='healthCode',how=how)

        # set dates
        self.parse_dates = ['startTime','endTime']
        self.usecols = usecols

        # backfill ids
        if self.include_all_other_patients:
            self.hktable['external_identifier'] = self.hktable['external_identifier'].fillna(self.hktable['healthCode'])
            self.sleeptable['external_identifier'] = self.sleeptable['external_identifier'].fillna(self.sleeptable['healthCode'])
            self.workouttable['external_identifier'] = self.workouttable['external_identifier'].fillna(self.workouttable['healthCode'])
            self.wtdf['external_identifier'] = self.wtdf['external_identifier'].fillna(self.wtdf['healthCode'])
            self.id_link = pd.concat([self.id_link,
                                      self.hktable.loc[:,['healthCode', 'external_identifier']],
                                      self.sleeptable.loc[:,['healthCode', 'external_identifier']],
                                      self.workouttable.loc[:,['healthCode', 'external_identifier']],
                                      self.wtdf.loc[:,['healthCode', 'external_identifier']],
                                      ],axis=0).drop_duplicates()


    def zip_table_like_path(self, table_name: str):
        
        """Returns a decoded zip path containing the requested table from the zip file if it exists.
        
        Args:
            table_name (str): The name of the table required

        Returns:
            Path: The path to the table's file
        """
        
        s = subprocess.run(f'unzip -l {self.base_path}/{self.zipname} | grep {table_name}',
                       check=True,capture_output=True,shell=True)
        if s.returncode != 0:
            raise FileNotFoundError(f"Table {table_name} not found in the zip file")

        zip_path = " ".join(s.stdout.decode().split()[3:])
        # return pd.read_csv(f'zip:/{self.base_path}/{self.zipname}!{zip_path}') 
        # zf = zipfile.ZipFile(f'{self.base_path}/{self.zipname}') 
        return self.zf.open(zip_path)

    def get_healthkit_data(self, external_id: str, no_cache: bool = False, limit: int = None,selected_table = 'activity') -> pd.DataFrame:
        """Returns the full healthkit dataset for the given external_id

        Args:
            external_id (str): The identifier to be filtered for
            no_cache (bool, optional): If True, will not cache the data or use a previously-saved cache. Defaults to
                                       False.
            limit (int, optional): If provided, limit only to this number of entries. Useful for testing.

        Returns:
            pd.DataFrame: The HealthKit dataset
        """
        # Make sure extid is capitalised
        external_id = external_id.upper()
    
        # # Load from cache if we can
        cache_file = (self.hkcache / external_id).with_suffix('.pickle')

        if not no_cache and cache_file.is_file():
            #print("IN")   #add Juan
            return pd.read_pickle(cache_file)

        # # Otherwise, load the healhkit table from the synapseCache if this hasn't been done yet
        # if not hasattr(self, 'hktable'):
        #     # self.hktable = self.load_table("cardiovascular-HealthKitDataCollector-v1")
        #     self.hktable = pd.read_csv(self.zip_table_like_path("cardiovascular-HealthKitDataCollector-v1"))
        
        # select table 
        if selected_table == 'activity':
            tablenow = self.hktable
        elif selected_table == 'sleep':
            tablenow = self.sleeptable
        elif selected_table == 'workout':
            tablenow = self.workouttable
        else:
            raise "No table has been selected, please select one of these: activity, sleep, workout"
        tablenow['external_identifier'] = tablenow['external_identifier'].str.upper()

        # Filter the healthkit table by external ID and grab data.csv column
        blobs = tablenow[tablenow.external_identifier == external_id]['data.csv']
        blobs = blobs.dropna().reset_index(drop=True)

        # Set limit if requested
        if limit:
            blobs = blobs[0:limit]

        # Concatenate all data.csv files for this ID
        data_csvs = (self.load_data(blob) for blob in blobs.astype(int).values)
        # data_csvs = (self.load_data(blob) for _, blob in blobs.iteritems())

        # Concatenate the results, wrapping our generator with tqdm for a progress bar
        df = pd.concat(tqdm(data_csvs, total=len(blobs), position=0, leave=True, desc=f"Loading healthkit {selected_table} data"),
                       ignore_index=True)

        # Ensure cache dir exists
        self.hkcache.mkdir(parents=True, exist_ok=True)

        # Save to cache and return
        if not no_cache:
            df.to_pickle(cache_file)
        return df

    def load_data(self, blob_id: str) -> pd.DataFrame:
        """Loads a dataset from a data.csv file

        Args:
            blob_id (str): The blob identifier for the data.csv

        Returns:
            pd.DataFrame: Data output
        """
        try:
        # if True:
            # There are some items in these files which have commas in an unquoted device name field.
            # For reference, a list of all possible device IDs can be found at:
            # https://gist.githubusercontent.com/adamawolf/3048717/raw/19b4cc0627ff669f1c4589324b9cb45e4948ec01/Apple_mobile_device_types.txt

            # The following regex will match any (current) Apple device ID that contains commas:
            # device_re = re.compile(r"(?:iPhone|iPad|iPad|Watch)\d+,\d+")

            # Function to surround with quotes any device IDs which contain commas
            # def fix_bad_device_ids(path):
            #     with open(path) as f:
            #         return StringIO('\n'.join(device_re.sub(r'"\g<0>"', row) for row in f))
            # is this needed? - how to solve?
            # def fix_bad_device_ids(f):
            #     return StringIO('\n'.join(device_re.sub(r'"\g<0>"', str(row)) for row in f))
            
            def manual_separation(bad_line):
                right_split = bad_line[:5] + [",".join(bad_line[5:7])] + bad_line[7:] # All the "bad lines" come from source
                return right_split
            
            # Read in the dataget_healthkit_data
            # data = pd.read_csv(fix_bad_device_ids(self.blob_path(blob_id)),
            # data = pd.read_csv(fix_bad_device_ids(self.zf.open(self.blob_path(blob_id))),

            # with self.zf.open(self.blob_path(blob_id,'data.csv')) as fid:
                # with open('data.csv', 'wb') as f:
                    # f.write(fid.read())
            data = pd.read_csv(self.zf.open(self.blob_path(blob_id,'.csv')),#'data.csv'
                            dtype={'value': np.float64},
                            parse_dates=self.parse_dates,
                            # date_format=lambda col: pd.to_datetime(col, utc=True, errors='coerce'),
                            quotechar='"',
                            usecols=self.usecols,
                            na_values=['value'],
                            on_bad_lines=manual_separation,
                            index_col=False,
                            engine='python')
            
            # this is an exception for workout data - not implemented
            # if data.shape[0] == 0: # if nothing loaded try clearning the metadata pattern
            #     from io import StringIO
            #     with self.zf.open(self.blob_path(blob_id,'.csv')) as fid:
            #         file = fid.read()
            #         pattern = r"{(.*?)}"

            #         # Replace the matched content with an empty string
            #         result = re.sub(pattern, "", file.decode('UTF-8'))
            #         data = pd.read_csv(StringIO(result))

            # calculate the duration
            if 'endTime' in self.parse_dates:
                tic = datetime.now()
                for i,row in data.iterrows():
                    data.loc[i,'duration'] = (pd.to_datetime(row['endTime'])-pd.to_datetime(row['startTime'])).total_seconds()/60
                # print("duration processed in:",datetime.now() - tic,'s')
            # Strip out any header rows that have been picked up
            return data[data.type != 'type']
        except Exception as e:
            print(e)
            return pd.DataFrame([])

    def blob_path(self, blob_id: str,ending: str) -> Path:
        """Get the path to a synapse cache blob by its ID

        This will return the path of the first non-hidden file found in the appropriate directory.

        Args:
            blob_id (str): The blob identifier

        Raises:
            FileNotFoundError: If there are no files found at the blob's directory

        Returns:
            Path: Path to the blob data
        """

        # Convert blob_id from int to str, just in case
        blob_id = str(blob_id)

        # Build the path
        # path = self.cache_path / str(int(blob_id[-3:])) / blob_id
        path = os.path.join(self.synapse_zip_path,str(int(blob_id[-3:])),blob_id)
        # UK_Export/synapseCache/814/19045814/data-8916be20-3dcf-42da-b6a1-277c1684a891.csv
        #print("blob_id:",path)     #add Juan

        # Look for all files containing data.csv
        try:
            # file_path = next(f for f in path.iterdir() if f.is_file() and not f.name.startswith('.'))
            file_path = next(f for f in self.zf.infolist() if f.filename.startswith(path) and f.filename.endswith(ending) and not f.is_dir()).filename
        except StopIteration:
            raise FileNotFoundError("Blob path contains no files")
        else:
            return file_path


    def get_6mwt_data(self, external_id: str) -> pd.DataFrame:
        """Gets the 6-minute walk test data for the given ID

        Args:
            external_id (str): The external ID to select data for

        Returns:
            pd.DataFrame: Resulting dataset
        """
        
        #print("wtdf: ", wtdf)   #add Juan
        current_wt = self.wtdf.external_identifier == external_id
        
        if not current_wt.any():
            print(f"No 6MWT data available for {external_id}")
            return pd.DataFrame([],columns=['startDate', 'endDate', 'numberOfSteps', 'distance', 'floorsAscended',
                                          'floorsDescended'])

        # Pull the json blob for each blob_id in the pedometer_fitness.walk column; we only want the last row in 'items'
        jsons = (self.load_zip_json(str(int(blob_id)))['items'][-1]
                 for blob_id in self.wtdf[current_wt]["pedometer_fitness.walk"].dropna())
        
        #for blob_id in wtdf[wtdf.external_identifier == external_id]["pedometer_fitness.walk"].dropna():
        #    print("blob_id in for:",blob_id)
        #    print("wtdf.external_identifier: ",wtdf.external_identifier)
        #    print("external_id: ",external_id)
            #print("wtdf.external_identifier==external_id:",wtdf.external_identifier == external_id)
            #print("wtdf:",wtdf[wtdf.external_identifier == external_id]["pedometer_fitness.walk"].dropna())
            
        # Create a DataFrame and put the columns in a consistent order
        df = pd.DataFrame(jsons, columns=['startDate', 'endDate', 'numberOfSteps', 'distance', 'floorsAscended',
                                          'floorsDescended'])

        # Turn startDate and endDate into datetime64s
        for col in ('startDate', 'endDate'):
            df[col] = pd.to_datetime(df[col], utc=True)

        # Index on startDate for convenience
        return df.set_index('startDate')


    def load_zip_json(self, blob_id: str) -> Any:
        """Load a JSON blob from a zipped file

        Args:
            blob_id (str): The blob identifier

        Returns:
            Any: The parsed JSON data as a Python object
        """
        try:
            with self.zf.open(self.blob_path(blob_id,'.walk')) as f:
                #print("path to load json:",self.blob_path(blob_id))   #add Juan
                return json.load(f)
        except Exception as e:
            print(e)
            return 

    def set_device(self,hk: pd.DataFrame)-> None:
        
        hk.loc[:,'device'] = ''
        hk['day'] = pd.to_datetime(hk['startTime'],utc=True).dt.strftime('%Y-%m-%d')

        # replace the name of the device
        for device in devices_to_keep:
            print(device)
            
            idx = hk['source'].str.lower().str.contains(device.lower())
            hk.loc[idx,'device'] = device

            a0 = hk.loc[idx,:].copy().drop(columns=['device_rank']) if 'device_rank' in hk.keys() else hk.loc[idx,:].copy()

            # identify the devices
            # aa = a0.groupby(['patient','source','device']).agg({'value':'count','startTime':['min','max']}).reset_index()
            # aa.columns = ['_'.join(col).strip('_') for col in aa.columns]
            # pdg.to_gbq(aa.astype(str),'MHC_PH.device_log',project_id=project_id,if_exists='append')

            # find primary watch 
            a = a0.groupby(['day','source'])['value'].count().reset_index().sort_values(by=['day','value'],ascending=[True,False])
            a['device_rank'] = device + (a.groupby('day').cumcount() + 1).astype(str)
            # merge
            out = a0.merge(a.drop(columns=['value']),on=['day','source'])
            hk.loc[idx,'device_rank'] = out['device_rank']
        
        return hk

    def correct_multilevel(self,aa):
        all_cols_but_value = [c for c in aa.columns.get_level_values(0) if c != 'value']
        d = []
        for value in aa['value'].keys():
            a2 = aa.loc[:,all_cols_but_value].copy()
            a2.columns = a2.columns.droplevel(1)
            a2['value'] = aa['value'][value]
            a2['type'] = a2['type'] + value.capitalize()
            d.append(a2.copy())
        return pd.concat(d,axis=0)

    def aggregate_hkdata(self,hk,date_format='%Y-%m-%d'):
        d = []
        hkaux = hk.copy()
        hkaux['startTime'] = pd.to_datetime(hkaux['startTime'],utc=True,format='mixed').dt.strftime(date_format)
        for gi,g in list(hkaux.groupby('type')):
            value_dict = {'value':aggregation_type[gi]}
            if 'duration' in hkaux.keys():
                value_dict.update({'duration':'sum'}) 
            if isinstance(value_dict['value'],list):  
                d.append(self.correct_multilevel(g.groupby(['patient','device','source','device_rank','type','startTime']).agg(value_dict).reset_index()))
            else:    
                d.append(g.groupby(['patient','device','source','device_rank','type','startTime']).agg(value_dict).reset_index())
        return pd.concat(d,axis=0)

    def add_pace(self,hk: pd.DataFrame)-> pd.DataFrame:
        # to add pace I am going to filter all instances where there was less than 30 seconds recorded
        hkaux = hk.query("type in ('HKQuantityTypeIdentifierStepCount','HKQuantityTypeIdentifierFlightsClimbed') and duration > 0.5")
        hkaux.loc[:,'value'] /= hkaux.loc[:,'duration']
        hkaux.loc[:,'type'] = hkaux.loc[:,'type'] + "Pace"
        hk = pd.concat([hk,hkaux],axis=0,ignore_index=True).reset_index(drop=True)
        print('pace added to data')
        try:
            hkaux.columns = [c.replace('[^\w\s]', '_') for c in hkaux.columns]
            hkaux = hkaux.astype(str)
            # this doesnt seem to work for a few patients
            # pdg.to_gbq(hkaux, "MHC_PH.raw_activity_pace_nodup", project_id=project_id, if_exists='append', api_method="load_csv")
            print('pace added to GBQ')
        except Exception as e:
            print(e)
            hkaux.to_csv(f'bkps/pace{hkaux["patient"].unique()[0]}.csv',index=False)
        return hk
    
    def add_cardiac_effort(self,hk: pd.DataFrame)-> pd.DataFrame:
        
        # identify the activity
        hkaux1 = hk.query("type == 'HKQuantityTypeIdentifierWalkingHeartRateAverage'")
        hkaux2 = hk.query("type == 'HKQuantityTypeIdentifierDistanceWalkingRunning'")#

        hkaux1.loc[:,'startTime_d'] = pd.to_datetime(hkaux1.loc[:,'startTime'],utc=True).dt.strftime('%Y-%m-%d')
        hkaux2.loc[:,'startTime_d'] = pd.to_datetime(hkaux2.loc[:,'startTime'],utc=True).dt.strftime('%Y-%m-%d')
        hkaux1.loc[:,'beats'] = hkaux1.loc[:,'value']*hkaux1.loc[:,'duration']

        # calculate beats/s average per day
        hkaux1a = hkaux1.groupby(['patient','device','startTime_d']).agg({'beats':'sum','duration':'sum'}).reset_index()
        hkaux1a['value'] = hkaux1a['beats']/hkaux1a['duration']

        # merge by day
        hkaux = hkaux1a.merge(hkaux2,on=['patient','device','startTime_d'],suffixes=('_aux',''))

        # then multiply by seconds of activity and divide by distance
        hkaux['value'] = hkaux['value_aux']*hkaux['duration']/hkaux['value']

        # drop all cols not needed
        hkaux = hkaux.drop(columns=['beats','duration_aux','value_aux','startTime_d'])
        hkaux.loc[:,'type'] = "HKQuantityTypeIdentifierCardiacEffort"
        hk = pd.concat([hk,hkaux],axis=0,ignore_index=True).reset_index(drop=True)
        print('cardiac effort added to data')

        return hk
    

def remove_duplicates(logger,Ext_ID,hk,data_name='activity'):
    s = hk.shape[0]
    hk.drop_duplicates(inplace=True)
    print(f"Removed {(s-hk.shape[0])} ({'%.0f'%((s-hk.shape[0])/s*100)}%) duplicates from the {data_name} data for {Ext_ID}.")
    logger.info(f"Removed {(s-hk.shape[0])} ({'%.0f'%((s-hk.shape[0])/s*100)}%) duplicates from the activity data for {Ext_ID}.")

    # remove duplicates by startTime too 
    s = hk.shape[0]
    hk.drop_duplicates(subset=['startTime','type','source'],inplace=True)   # we included value here before - now removing all - overlaps with endTime are not considered here
    if 'endTime' in hk.keys():
        hk.drop_duplicates(subset=['endTime','type','source'],inplace=True)   # the same logic is true for the end times
    print(f"Removed {(s-hk.shape[0])} ({'%.0f'%((s-hk.shape[0])/s*100)}%) STARTTIME duplicates from the {data_name} data for {Ext_ID}.")
    logger.info(f"Removed {(s-hk.shape[0])} ({'%.0f'%((s-hk.shape[0])/s*100)}%) STARTTIME duplicates from the activity data for {Ext_ID}.")


# for ii,Ext_ID in enumerate(list_Ext_IDs_All):
    # Ext_ID = 'SPVDU07120'
# if True:
    # Ext_ID = 'SPVDU03163'
    # if Ext_ID == 'SPVDU07075':
    #     continue
    # start = Ext_ID == 'SPVDU01264' if not start else True
    # if not start:
    #     continue

def complete_loop(Ext_ID,logger,sc,suffix,tables_to_load):
    # Ext_ID = 'SPVDU00165'
    print(f"\nLoading Healthkit data for ID: {Ext_ID} |||||||||||...Loading...|||||||||||\n")
    for tablenow in tables_to_load:    
        try_small_loop(Ext_ID,logger,sc,suffix,tablenow)

def try_small_loop(Ext_ID,logger,sc,suffix,tablenow,if_exists='append'):    
    try:
    # if True:
        # "Obtain data of 6-minute walk activity and save .csv"
        # if '6mwt' in tables_to_load:
        if '6mwt' == tablenow:
            print(f'{Ext_ID} 6mwt started...')
            dfwt = sc.get_6mwt_data(Ext_ID)
            dfwt['patient'] = Ext_ID
            
            # remove duplicates
            dfwt.drop_duplicates(inplace=True)

            pdg.to_gbq(dfwt, f"MHC_PH.6mwt{suffix}", project_id=project_id, if_exists=if_exists)
            print(f'{Ext_ID} 6mwt uploaded to GBQ')

        # get activity data
        # if 'activity' in tables_to_load:
        if 'activity' == tablenow:
            print(f'{Ext_ID} activity started...')
            # this is a record of what has been attempted
            # attempted = pd.DataFrame(np.array([datetime.now().strftime('%Y-%m-%d'),Ext_ID]).reshape(-1,2),columns=['date','patient'])
            # pdg.to_gbq(attempted, "MHC_PH.attempted", project_id=project_id, if_exists='append')

            sc.parse_dates = ['startTime','endTime']
            sc.usecols = None
            hk = sc.get_healthkit_data(Ext_ID,no_cache=True,selected_table = 'activity')
            hk['patient'] = Ext_ID

            # remove duplicates
            remove_duplicates(logger,Ext_ID,hk,data_name='activity')

            # set device type
            hk = sc.set_device(hk)
            
            # add pace
            hk = sc.add_pace(hk)

            # add cardiac effort
            hk = sc.add_cardiac_effort(hk)
    
            # # save devices
            # aux = hk.loc[:,['patient','device','source']].drop_duplicates()
            # pdg.to_gbq(aux, "MHC_PH.devices", project_id=project_id, if_exists=if_exists)
            # filter by devices allowed - I am going to allow them all
            # hk = hk.loc[hk['device'].str.lower().isin(device_type),:].reset_index(drop=True)
            # hk = hk.dropna(subset=["device"]).reset_index(drop=True)
            idx = hk['device'] == ''
            hk.loc[idx,'device'] = hk.loc[idx,'source'] # this allows them all

            # save the raw by hourly data
            hk_hour = sc.aggregate_hkdata(hk,date_format='%Y-%m-%d-%H')
            pdg.to_gbq(hk_hour, f"MHC_PH.raw_activity_hour{suffix}", project_id=project_id, if_exists=if_exists)
            print(f'{Ext_ID} activity (hourly) uploaded to GBQ')
            del hk_hour

            # aggregate the data
            hk_out = sc.aggregate_hkdata(hk,date_format='%Y-%m-%d')
            # save
            if save_local:
                hk_out.to_csv(f'bkps/{Ext_ID}.csv',index=False)
            else:
                pdg.to_gbq(hk_out, f"MHC_PH.raw_activity{suffix}", project_id=project_id, if_exists=if_exists)
                print(f'{Ext_ID} activity uploaded to GBQ')
            
            # # save the units
            # aux = hk.loc[:,["patient","type","unit"]].drop_duplicates()
            # pdg.to_gbq(aux, "MHC_PH.units", project_id=project_id, if_exists=if_exists)
            # print('units activity sunk uploaded to GBQ')

        # get workout data
        # if 'workout' in tables_to_load:
        if 'workout' == tablenow:
            print(f'{Ext_ID} workout started...')
            sc.parse_dates = ['startTime','endTime']
            sc.usecols = None#["startTime","endTime","type","workoutType","total distance","unit","energy consumed","unit","source","sourceIdentifier","metadata","appVersion"]
            workout = sc.get_healthkit_data(Ext_ID,no_cache=True,selected_table = 'workout')
            workout['patient'] = Ext_ID
            sc.set_device(workout)
            # workout = workout.loc[workout['device'].isin(device_type),:].reset_index(drop=True)
            workout['type'] = workout['workoutType']
            workout = workout.melt(id_vars=['patient','device','startTime','endTime','type','unit'],value_vars = ['total distance','energy consumed'])
            workout['variable'] = workout['variable'].str.replace(" ","_")
            workout = workout.query('type!="(null)"').reset_index(drop=True)
            workout['type'] = workout['type'] + workout['variable']
            workout['startTime'] = pd.to_datetime(workout['startTime'],utc=True).dt.strftime('%Y-%m-%d')
            
            # remove duplicates
            remove_duplicates(logger,Ext_ID,workout,data_name='workout')

            # set device type
            workout = sc.set_device(workout)

            workout_out = sc.aggregate_hkdata(workout)

            pdg.to_gbq(workout_out, f"MHC_PH.raw_workout{suffix}", project_id=project_id, if_exists=if_exists)
            print(f'{Ext_ID} workout uploaded to GBQ')
            # aux = workout.loc[:,["patient","type","unit"]].drop_duplicates()
            # pdg.to_gbq(aux, "MHC_PH.units", project_id=project_id, if_exists='append')
            # print('units workout sunk uploaded to GBQ')

        # get sleep data
        # if 'sleep' in tables_to_load:
        if 'sleep' == tablenow:
            print(f'{Ext_ID} sleep started...')
            sc.parse_dates = ['startTime']
            sc.usecols = None
            sleep = sc.get_healthkit_data(Ext_ID,no_cache=True,selected_table = 'sleep')
            sc.set_device(sleep)
            cols_sleep = ['startTime', 'category value', 'value', 'source','device','patient']
            sleep['patient'] = Ext_ID
            sleep_sel = sleep.loc[:,cols_sleep]
            sleep_sel.rename(columns={'category value':'type'},inplace=True)
            
            # remove duplicates
            remove_duplicates(logger,Ext_ID,sleep_sel,data_name='sleep')

            # set device type
            sleep_sel = sc.set_device(sleep_sel)
            
            # aggregate the data
            sleep_out = sc.aggregate_hkdata(sleep_sel)

            # ['patient','device','type','startTime'] (sleep_out.groupby('type')['value']).describe()/3600
            pdg.to_gbq(sleep_out, f"MHC_PH.raw_sleep{suffix}", project_id=project_id, if_exists=if_exists)
            print(f'{Ext_ID} sleep uploaded to GBQ')

            # aux = sleep.loc[:,["patient","type","unit"]].drop_duplicates()
            # pdg.to_gbq(aux, "MHC_PH.units", project_id=project_id, if_exists='append')
            # print('units sleep uploaded to GBQ')
        
        # for testing
        # hk.to_csv("../data/raw/healthkit.csv",index=False)
        # pdg.to_gbq(df,"MHC_PH.rawdata", project_id=project_id,if_exists=if_exists)
        # pdg.to_gbq(hk, "MHC_PH.rawdata_new", project_id=project_id, if_exists='append')
        # hk = pd.read_csv("../data/raw/healthkit.csv")
        # hk['patient'] = Ext_ID

    except Exception as e:
        logger.error(f"Loading FAILED!!! The External ID: {Ext_ID} had an error!")
        logger.error(str(e))
        print(e)
        print(f"Loading FAILED!!! The External ID: {Ext_ID} had an error!")
        if 'hk' in locals():
            hk.to_csv('bkps/'+Ext_ID+'bkp.csv',index=False)        

# json_data = pd.Series(aux.unit.values, index=aux.type).to_json()
        # with open(f'../data/raw/{Ext_ID}_units.json', 'w') as file:
        #     file.write(json_data)
        
