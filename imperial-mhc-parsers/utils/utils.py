import pandas as pd
import numpy as np
import pandas_gbq as pdg
from utils.constants import *
from google.cloud import bigquery
import os

def delete_tables(table,suffix):
    
    try:
        bqclient = bigquery.Client()

        query = bqclient.query(f"delete from `{project_id}.MHC_PH.raw_{table}{suffix}` WHERE TRUE")
        print("started query as {}".format(query.job_id))
        # invoke result() to wait until completion
        # DDL/DML queries don't return rows so we don't need a row iterator
        query.result()
        print(f"`{project_id}.MHC_PH.raw_{table}{suffix}` deleted")
        
    except Exception as e:
        print(e)

def find_outliers(df):
    D = []
    for di,df2 in df.groupby(['device','aggregation']):
        dfp = df2.pivot(index='startTime',columns='variable', values='value')

        outliers = dfp-(dfp.mean() + 1.96*dfp.std())>0
        cols = outliers.columns
        outliers = outliers.reset_index()
        outliers = outliers.melt(id_vars="startTime",value_vars=cols)
        outliers = outliers.rename(columns={"value":'outlier'})
        outliers['device'] = di[0]
        outliers['aggregation'] = di[1]
        D.append(df2.merge(outliers,on=['startTime', 'variable','device','aggregation']))
    return pd.concat(D,axis=0,ignore_index=True)

def calculate_dates(df):
    df["startTime"] = pd.to_datetime(df["startTime"])
    g = df.groupby("patient")["startTime"]
    aux = pd.concat([g.min(),
                    g.max()],
                    axis=1)
    aux.columns = ["min","max"]
    aux.reset_index(inplace=True)
    df = df.merge(aux,on="patient",how='left')
    df["days"] = (pd.to_datetime(df["startTime"]) - pd.to_datetime(df["min"])).dt.days
    df["duration_days"] = (pd.to_datetime(df["max"]) - pd.to_datetime(df["min"])).dt.days
    df["duration_months"] = df["duration_days"]//30
    df.drop(columns=["min","max"],inplace=True)
    return df

def calculate_sleep_ratios(df):
    inbed = df.query('variable == "InBed"').rename(columns={'value':'inbed'})
    awake = df.query('variable == "Asleep"').loc[:,['patient','startTime','value']]
    asleep = df.query('variable == "Awake"').loc[:,['patient','startTime','value']]
    awake = inbed.merge(awake,on=['patient','startTime'],how='inner')
    asleep = inbed.merge(asleep,on=['patient','startTime'],how='inner')
    awake['value'] = awake['value']/awake['inbed']
    asleep['value'] = asleep['value']/asleep['inbed']
    awake['variable'] = 'Awake ratio'
    asleep['variable'] = 'Asleep ratio'
    awake['type'] = 'Awake ratio'
    asleep['type'] = 'Asleep ratio'
    return pd.concat([df,awake.drop(columns=['inbed']),asleep.drop(columns=['inbed'])],axis=0,ignore_index=True)

class Raw_ETL():
    def __init__(self,df,dfm,sink = True,suffix=""):
        self.df = df
        self.dfm = dfm
        self.sink = sink
        self.suffix = suffix

    def run_clean_pipeline(self):
        # transform the date
        self.df['variable'] = self.df['type'].str.replace('HKQuantityTypeIdentifier','').str.replace('HKCategoryValue','').str.replace('SleepAnalysis','')

        # date transformation
        self.df["startTime"] = pd.to_datetime(self.df["startTime"]).dt.strftime('%Y-%m-%d')

        # get value in common units
        self.df['unit_conversion'] = self.df['variable'].map(unit_conversion)
        self.df['value_orig'] = self.df['value'].copy()
        self.df['value'] *= self.df['unit_conversion']

        # clean device
        idx = self.df['device'].str.lower().str.contains('phone').fillna(False)
        self.df.loc[idx,'device'] = 'iPhone'
        idx = self.df['device'].str.lower().str.contains('watch').fillna(False)
        self.df.loc[idx,'device'] = 'Watch'

        # calculate dates
        self.df = calculate_dates(self.df)

        # add ids
        self.df = self.df.merge(self.dfm,on="patient",how='left')
        self.df["Group"] = self.df["Group"].fillna("Not found")

        self.df["imputation"] = "None"
        self.df["clean_status"] = "raw"
        # add cohort
        self.df['cohort'] = self.df['patient'].apply(lambda x:'UK' if x.startswith('SPVDU') else 'US')

        # sink the data
        if self.sink:
            pdg.to_gbq(self.df,f"MHC_PH.activity{self.suffix}", project_id=project_id,if_exists='replace')
        
class ETL():
    def __init__(self,df,sink=True,suffix=""):
        self.df_clean = df.copy()
        self.p = {'remove_nans':0,'remove_dates':0,'remove_devices':0,'remove_vars':0,'remove_lower':0,'remove_upper':0}
        self.keys = df.keys()
        self.sink = sink
        self.suffix = suffix

    def run_clean_pipeline(self):

        # run each cleaning step
        self.remove_oos_devices()
        self.map_devices()
        self.remove_oos_vars()
        self.remove_bounds()
        self.remove_nans()
        self.remove_dates()
        
        # add other variables
        self.add_heartratereserve()
        self.add_bmi()
        self.add_bedbound()
        self.add_cardiaceffort()
        self.add_time_over_effort_threshold(var='FlightsClimbed',pct_obj = .7)
        self.add_time_over_effort_threshold(var='StepCount',pct_obj = .7)

        # add additional fields
        self.add_support_fields()

        if self.sink:
            self.sink_data()

    def remove_oos_devices(self):
        # data cleaning steps
        # 1. remove unwanted devices
        s = self.df_clean.shape[0]
        self.df_clean = self.df_clean.query(f'device in {devices_to_keep}').reset_index(drop=True)
        self.p['remove_devices'] += s - self.df_clean.shape[0]
        print('OOS devices:',s - self.df_clean.shape[0],'rows -','%.1f'%((s - self.df_clean.shape[0])/s*100),'%')
    
    def map_devices(self):
        # we are mapping some devices
        self.df_clean['device'] = self.df_clean['device'].map(device_mapping)
        print('Devices re-mapped!')
    
    def remove_oos_vars(self):
        # 2. remove unwanted variables
        s = self.df_clean.shape[0]
        self.df_clean = self.df_clean.query(f'variable not in ({to_drop})').reset_index(drop=True)
        self.p['remove_vars'] += s - self.df_clean.shape[0]
        print('OOS variables:',s - self.df_clean.shape[0],'rows -','%.1f'%((s - self.df_clean.shape[0])/s*100),'%')
    
    def remove_bounds(self):
        # 3. remove data out of bounds
        self.df_clean['min_bound'] = self.df_clean['variable'].map(bound_min)
        self.df_clean['max_bound'] = self.df_clean['variable'].map(bound_max)

        s = self.df_clean.shape[0]
        self.df_clean = self.df_clean.query('value >= min_bound').reset_index(drop=True)
        self.p['remove_lower'] += s - self.df_clean.shape[0]
        print('Below lower bound:',s - self.df_clean.shape[0],'rows -','%.1f'%((s - self.df_clean.shape[0])/s*100),'%')

        s = self.df_clean.shape[0]
        self.df_clean = self.df_clean.query('value <= max_bound').reset_index(drop=True)
        self.p['remove_upper'] += s - self.df_clean.shape[0]
        print('Above upper bound:',s - self.df_clean.shape[0],'rows -','%.1f'%((s - self.df_clean.shape[0])/s*100),'%')

    def remove_nans(self):
        # 4. remove nans
        s = self.df_clean.shape[0]
        self.df_clean = self.df_clean.dropna(subset=['value']).reset_index(drop=True)
        self.p['remove_nans'] += s - self.df_clean.shape[0]
        print("Nans removed:",self.p['remove_nans'],'rows -','%.1f'%((s - self.df_clean.shape[0])/s*100),'%')

    def remove_dates(self):
        # 5. remove illogical dates
        s = self.df_clean.shape[0]
        self.df_clean = self.df_clean.query('startTime > "2000-01-01"').reset_index(drop=True)
        self.p['remove_dates'] += s - self.df_clean.shape[0] 
        print("Illogical dates removed:",self.p['remove_dates'],'rows -','%.1f'%((s - self.df_clean.shape[0])/s*100),'%')

    def add_support_fields(self):
        # add the status
        self.df_clean["clean_status"] = "clean_1"
        # relaculate dates
        self.df_clean = calculate_dates(self.df_clean)
        # add sleep calculation        
        self.df_clean = calculate_sleep_ratios(self.df_clean)

    def sink_data(self):        
        # sink the data
        self.df_clean = self.df_clean.loc[:,self.keys]
        pdg.to_gbq(self.df_clean,f"MHC_PH.activity{self.suffix}", project_id=project_id,if_exists='append')
        print('Clean_1 data added!')

    def add_heartratereserve(self):
        # Heart rate reserve – resting to your avg
        df2 = self.df_clean.query("variable in ('HeartRate','RestingHeartRate') and device == 'Watch'")
        df3 = df2.pivot_table(index=["patient","startTime"],columns="variable",values="value",aggfunc="mean").reset_index()
        df3 = df3.dropna(subset=["HeartRate","RestingHeartRate"],inplace=False).reset_index(drop=True)
        df3["HeartRateReserve"] = df3["HeartRate"] - df3["RestingHeartRate"]
        # melt the data
        df3 = df3.melt(id_vars=["patient","startTime"],value_vars=["HeartRateReserve"],value_name="value",var_name="variable")
        vars_to_add = [k for k in df2.keys() if k not in df3.keys()]+["patient","startTime"]
        df3 = df3.merge(df2.query('variable=="HeartRate"').drop_duplicates(subset=['patient','startTime']).loc[:,vars_to_add],on=["patient","startTime"],how='inner')
        # sink the data
        # pdg.to_gbq(df3.loc[:,self.keys],"MHC_PH.activity", project_id=project_id,if_exists='append')
        self.df_clean = pd.concat([self.df_clean,df3],axis=0,ignore_index=True)
        print('HeartRateReserve added!')
            
    def add_bmi(self):
        # BMI
        df_bmi = self.df_clean.query("variable in ('Height','BodyMass')")
        df_bmi = df_bmi.groupby(["patient","variable"])["value"].mean().reset_index()
        df_bmi = df_bmi.pivot(index="patient",values="value",columns="variable")
        # df_bmi["Height_inferred"] = df_bmi["Height"].isna() 
        df_bmi["Height"] = df_bmi["Height"]#.fillna(1.7)
        df_bmi["bmi"] = df_bmi["BodyMass"]/(df_bmi["Height"]**2)
        df_bmi["bmi_cat"] = pd.cut(df_bmi["bmi"], [0,18.5,24.9,29.9,np.inf], labels=["Underweight","Normal","Overweight","Obese"])
        df_bmi = df_bmi.reset_index()
        # df_bmi = df_bmi.melt(id_vars=["patient","Height_inferred"],value_vars=["bmi"],value_name="value",var_name="variable")
        df_bmi = df_bmi.melt(id_vars=["patient"],value_vars=["bmi"],value_name="value",var_name="variable")

        # add cols 
        cols = [k for k in self.df_clean.keys() if k not in df_bmi.keys()]
        df_bmi = df_bmi.merge(self.df_clean.query("variable == 'Height'").drop_duplicates(subset=['patient']).loc[:,cols+['patient']],on="patient",how="inner")
        # df_bmi["startTime"] = None

        # sink the data
        self.df_clean = pd.concat([self.df_clean,df_bmi],axis=0,ignore_index=True)
        # pdg.to_gbq(df_bmi.loc[:,self.keys],"MHC_PH.activity", project_id=project_id,if_exists='append')
        print('BMI added!')
    
    def add_bedbound(self,threshold_hrs=18):
        # Heart rate reserve – resting to your avg
        df2 = self.df_clean.query("variable == 'InBed'")
        # df2['value'] = (df2['value'] > threshold_hrs).astype(int)# this is the binary option
        df2.loc[:,'value'] -= threshold_hrs
        df2.loc[:,'value'] = np.clip(df2['value'],0,24)
        df2.loc[:,'variable'] = 'BedBound'
        self.df_clean = pd.concat([self.df_clean,df2],axis=0,ignore_index=True)
        print('BedBound added!')
    
    def add_cardiaceffort(self):
        WalkingHeartRateAverage = self.df_clean.query("variable == 'WalkingHeartRateAverage'").copy().rename(columns={'value':'WalkingHeartRate'})
        DistanceWalkingRunning = self.df_clean.query("variable == 'DistanceWalkingRunning'").copy().rename(columns={'value':'DistanceWalkingRunning'})
        df = WalkingHeartRateAverage.merge(DistanceWalkingRunning.loc[:,['patient','startTime','device','DistanceWalkingRunning']],on=['patient','startTime','device'],how='inner')
        df['value'] = df['WalkingHeartRate']/df['DistanceWalkingRunning']
        df.drop(columns=['WalkingHeartRate','DistanceWalkingRunning'],inplace=True)
        df.loc[:,'variable'] = 'CardiacEffort'
        self.df_clean = pd.concat([self.df_clean,df],axis=0,ignore_index=True)
        print('CardiacEffort added!')
    
    def add_time_over_effort_threshold(self,var='FlightsClimbed',pct_obj = .7):
        aux0 = self.df_clean.query(f"variable in ('{var}PaceMax')")
        aux = aux0.pivot_table(index=['patient','device'],values=['value'],columns='variable',aggfunc='max').reset_index()
        aux.columns = aux.columns.map('_'.join).str.strip('_')
        aux2 = self.df_clean.query(f"variable in ('{var}PaceMean')").merge(aux,on=['patient','device'])
        aux2['value'] = aux2['value'] >= aux2[f'value_{var}PaceMax']*pct_obj
        g = aux2.groupby(['patient','device'])['value']
        aux2 = (g.sum()/g.count()).reset_index()
        aux2['variable'] = f'{var}PaceHigher{int(pct_obj*100)}pct'
        out = aux0.drop(columns=['value','variable']).merge(aux2,on=['patient','device'])
        self.df_clean = pd.concat([self.df_clean,
                                out],axis=0)
        print(f'{var}PaceHigher{int(pct_obj*100)}pct added!')
        # print(out.groupby('Group')['value'].describe())
