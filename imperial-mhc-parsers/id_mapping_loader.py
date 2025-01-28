import pandas as pd
import pandas_gbq as pdg
import json
import os
import re
import sys
sys.path.append("/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts")
from utils.constants import *
from datetime import datetime,date
import numpy as np

plot_flag = True

# with open("../creds/sharepoint_creds.json") as fid:
#     creds = json.load(fid)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path 

# read IDs - old one
# dfm = pd.read_csv("/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/clinical/MHC_IDs.csv")
# dfm.loc[:,"InMHC"] = dfm.loc[:,"InMHC"].fillna(False).astype(bool)
# dfm.loc[:,"Withdrawn"] = dfm.loc[:,"Withdrawn"].fillna("N/A").astype(str)##### implement data integrity report
# dfm.loc[:,"date_file"] = "2023-01-26"

dfm = pd.read_excel("../data/clinical/ids/MasterMHC_IDs_Dx_Mar2024_last.xlsx",
                  sheet_name="2024_01_MHC IDs_Dx_dates")
dfm2 = pd.read_excel("../data/clinical/ids/MasterMHC_IDs_Dx_Mar2024_last.xlsx",
                  sheet_name="Alternative diagnosis dates")

dfus = pd.read_csv('../data/clinical/ids/2024_us_cohort.csv')

# dfus_db = pdg.read_gbq('''SELECT distinct patient healthCode FROM `imperial-410612.MHC_PH.raw_activity_us` 
#                        where patient not like "SPVDU%"
#                        group by patient
#                         having count(*) > 200''',project_id=project_id)

# dfus_db.merge(dfus,on='healthCode').groupby('group').count()


dfm_mapping = {'MHC_ID':'patient',
        'MHC_date of consent':'consent_date',
        'Date of Death':'death_date',
        'Deceased':'dead',
        'Hospital':'hospital',
        'Withdrawn':'Withdrawn', 
        'roottable_ethnicorigin_item':'ethnicity',
        'Comparison1':'Group',
        'Comparison2':'Group_simple', 
        'roottable_subcategory_ifx_description_text':'Group_subcategory',
        'roottable_daignosis_date':'diagnosis_date'} 

sheffield_ethnicity_dict = {'1. White British':"White British", 
                            "nan":"Unknown",
                             '.   Asian Other':"Asian Other",
                             'Indian':"Asian Other",
                              '.   White European':"White Other",
                              '3. White Other':"White Other",
                                '10. Other':"Other", '7. Black Caribbean':"Black Caribbean"}

dfm = dfm.loc[:,dfm_mapping.keys()].rename(columns=dfm_mapping)
dfm['ethnicity'] = dfm['ethnicity'].astype(str).map(sheffield_ethnicity_dict).fillna("Unknown")
dfm['dead'] = (dfm['dead'].astype(float) == 1.).astype(bool).fillna(False).astype(str)
dfm['Withdrawn'] = (dfm['Withdrawn'].astype(float) == 1.).astype(bool).fillna(False).astype(str)
dfm.loc[:,"date_file"] = datetime.now().strftime("%Y-%m-%d")

# merge US data
dfus = dfus.rename(columns={'healthCode':'patient','group':'Group'})
usmapping = {'healthy':'Healthy', 'pah':'PAH', 'other_vascular':'CV'}
dfus['Group'] = dfus['Group'].map(usmapping)
dfm = pd.concat([dfm,dfus],axis=0)

aux = dfm.merge(dfm2.loc[:,['AL Diagnosis','AL consent', 'patient']],on='patient',how='left')
aux['diagnosis_date'] = aux['AL Diagnosis'].fillna(aux['diagnosis_date'])
aux['consent_date'] = aux['AL consent'].fillna(aux['consent_date'])
aux.drop(columns=['AL Diagnosis','AL consent'],inplace=True)
pdg.to_gbq(aux,"MHC_PH.id_mapping2",project_id=project_id,if_exists='replace')

# questionnaire
paths = ['/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/clinical/questionnaire/QuestionKey.csv',
        '/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/clinical/questionnaire/MHQ_long_Sep24.csv']
a = pd.read_csv(paths[0])
b = pd.read_csv(paths[1])
b['external_identifier'] = b['external_identifier'].fillna(b['healthCode'])
b["Question"] = b["Question"].apply(lambda x:x.replace('_decoded',''))
b['invalid'] = b['value'].str.contains('(\[[.+]?\])')
b['monthInStudy'] = b['dayInStudy']//30
b['negative'] = b['value'].apply(lambda x:x in ('No', 'not at all', 'FALSE','None of the above'))
b['question_cat'] = b['Question'].map(question_cat)
b = b.rename(columns={'external_identifier':'patient'})
b = b.merge(dfm,on='patient',how='left')
# clean values
# alcohol exception 
idx = b.query('Question == "alcohol"').index
b.loc[idx,'value'] = b.loc[idx,'value'].str.replace("[","").str.replace("]","")
b['value_clean'] = b['value'].str.replace(r'\[\d+\]','', regex=True)
b0 = b.copy()

# tobacco exception
idx = b.query('Question == "tobaccoProductsEver"').index
b.loc[idx,'value_clean'] = (b.loc[idx,'value'] == '8').map({True:'Tobacco Never',False:'Tobacco Yes'})
# Cannabis exception
idx = b.query('Question in ("cannabisSmoking","cannabisVaping")').index
b.loc[idx,'value_clean'] = b.loc[idx,'value'].map({'No, I have never':'Cannabis Never','No, but I have in the past':'Cannabis In the past'}).fillna('Cannabis Yes')

onehot = ["education","ethnicity","heart_disease",'race',"vascular",
          "regionInformation.countryCode","NonIdentifiableDemographics.patientFitzpatrickSkinType","NonIdentifiableDemographics.patientBloodType"]# formerly also "methodQuitSmokeless","labwork",

extra_to_onehot = []
mydict = {}
for li,l in list(b.groupby('Question')):
    # if li == 'regionInformation.countryCode':

    # numeric - integers
    if l['value_clean'].str.isnumeric().all(): # this is only numbers
        b.loc[b['Question']==li,'value_clean'] = l['value_clean'].astype(int)
        print(li,'numeric integer')
        continue
    
    # numeric - float
    if l['value_clean'].str.contains('^\d*(?:\.\d+)?$').all(): # this also include decimals
        b.loc[b['Question']==li,'value_clean'] = l['value_clean'].astype(float)
        print(li,'numeric float')
        continue

    # imperative conversion
    if li in quest_conversion.keys():
        b.loc[b['Question']==li,'value_clean'] = l['value_clean'].map(quest_conversion[li]) 
        print(li,'imperative')
        continue

    # onehot encode
    if li in onehot:
        print(li,'onehot skipping...')
        continue
    
    # boolean
    if 'TRUE' in l['value_clean'].unique() or 'FALSE' in l['value_clean'].unique() or True in l['value_clean'].unique() or False in l['value_clean'].unique() or 'No' in l['value_clean'].unique() or 'Yes' in l['value_clean'].unique():# boolean
        b.loc[b['Question']==li,'value_clean'] = l['value_clean'].apply(lambda x:x in ('TRUE','Yes',True)).astype(int)
        print(li,'boolean')
        continue
    
    # boolean second
    if 'None of the above' in l['value_clean'].unique() or 'not at all' in l['value_clean'].unique():
        b.loc[b['Question']==li,'value_clean'] = l['value_clean'].apply(lambda x:not x in ('None of the above','not at all')).astype(int)
        print(li,'boolean2')
        continue

    # aggregated conversion
    if np.any([k in l['value_clean'].unique() for k in agg_map.keys()]):
        b.loc[b['Question']==li,'value_clean'] = l['value_clean'].map(agg_map)
        print(li,'aggregated')
        continue
    b.loc[b['Question']==li,'value_clean'] = None
    print(li,'has not been converted')
    extra_to_onehot.append(li)

# onehot encode - education
def remove_dup_education(baux):
    degreemap = {'Doctoral Degree (Ph.D., M.D., J.D., etc.) ':1,
                "Master's Degree":2,
                'High school diploma':5, 
                'College graduate or Baccalaureate Degree':3,
                'Some college or vocational school or Associate Degree':4,
                'Grade school':6
                }
    degreemapinv = {v:k for k,v in degreemap.items()}
    baux['value_clean'] = baux['value_clean'].map(degreemap)
    baux = baux.sort_values(by=['value_clean']).drop_duplicates(subset=['patient'],keep='first')
    baux['value_clean'] = baux['value_clean'].map(degreemapinv)
    return baux

# onehot encode
b2 = pd.DataFrame()
for ki,k in enumerate(onehot):
    baux = b.query(f"Question == '{k}'").reset_index(drop=True)
    k = k.replace('NonIdentifiableDemographics.','').replace('regionInformation.','')
    # remove duplicates
    if k == 'education':
        baux = remove_dup_education(baux)
    elif k in ('heart_disease','vascular'):  
        baux['value_clean'] = baux['value_clean'].apply(lambda x:str(x).split(', and '))#.replace(' and','')
        baux = baux.explode('value_clean')
    else: 
        baux.drop_duplicates(subset=['patient'],inplace=True)
    baux2 = pd.get_dummies(baux['value_clean'],prefix=k)
    baux2.index = baux.loc[:,'patient']
    if k in ('heart_disease','vascular'):
        baux2 = baux2.groupby(level=0).any()
    b2 = pd.concat([b2,baux2],axis=1)
    print(k,'done!!')

# b2.drop(columns=['None of the above'],inplace=True)
b2.rename(columns={'vascular_8':'vascular_PAH'},inplace=True)
# b2 = b2.groupby('patient').any().reset_index()

# now modify the questions clean - features
idx = b.query(f"Question not in {onehot}").index
b3 = b.loc[idx,:].reset_index(drop=True)
b3['value_clean'] = b3['value_clean'].astype(float)
b3 = b3.dropna(subset=['value_clean'])
b3 = b3.groupby(['patient','question_cat','Question','Group'])[['value_clean']].max().reset_index()
b3 = b3.pivot_table(index = 'patient',columns='Question',values=['value_clean']).reset_index().droplevel(0,axis=1)
b3.rename(columns={'':'patient'},inplace=True)
b = b3.merge(b2,on='patient',how='outer')
# transform all boolean into 1,0
b.loc[:,'alcohol':] = b.loc[:,'alcohol':].astype(float)
b.loc[:,'alcohol':] = b.loc[:,'alcohol':].fillna(-1)
b.loc[:,'alcohol':] = b.loc[:,'alcohol':].astype(int)

# modify column names for upload to gbq
b.columns = [c.replace('.','').strip().replace(',','').replace(')','').replace('/','').replace('(','').replace(' ','_') for c in b.columns]
b = b.astype(str)

# merge Sheet but correct the names - many have been curated manually
a = a.rename(columns={'Identifier':'Question'})
a2 = a.loc[:,['Sheet','Question']]
b0 = b0.merge(a2,on='Question',how='left')
a2['Question'] = a2['Question'].str.replace('NonIdentifiableDemographics.','').str.replace('regionInformation.','')

# map to new questions
A = []
for k in b.iloc[:,1:].keys():
    aux = a2.query(f'Question == "{k}"')
    if len(aux)==0:
        aaux = [oneh.replace('NonIdentifiableDemographics.','').replace('regionInformation.','') for oneh in onehot if oneh.replace('NonIdentifiableDemographics.','').replace('regionInformation.','') in k]
        aux = a2.query(f'Question == "{aaux[0]}"')
        aux.loc[:,'Question'] = k
    A.append(aux)
    print(k)
a2 = pd.concat(A,axis=0,ignore_index=True)
    
# save all
pdg.to_gbq(b0.astype(str),'MHC_PH.questionnaire_us2',project_id=project_id,if_exists='replace')
pdg.to_gbq(b,'MHC_PH.questionnaire_clean_us2',project_id=project_id,if_exists='replace')
pdg.to_gbq(a,'MHC_PH.questionnaire_key_us2',project_id=project_id,if_exists='replace')
pdg.to_gbq(a2,'MHC_PH.questionnaire_key_clean_us2',project_id=project_id,if_exists='replace')

# remove the odd one out - done manually

# dfm.loc[:,"InMHC"] = dfm.loc[:,"InMHC"].astype(str)
# pdg.to_gbq(dfm,"MHC_PH.id_mapping", project_id="imperial-410612",if_exists='replace')

####################### Clinical data
cols_final = ['patient', 'date_of_consent', 'consent_withdrawn', 'inMHC', 'dead',
       'gender', 'date_diagnosis', 'test_date', 'variable', 'value', 'filename', 
       'comorbidity', 'TriphicCode', 'age', 'bmi', 'ethnicity',# unique to imperial
       'hospital','date_loaded', 
       'value_cat', 'value_date']

# read clinical data
D = {}
for root,dirs,files in os.walk("../data/clinical"):
    if root.split(r'/')[-1] == "ids":
        continue
    for file in files: 
        if file.endswith(".csv"):
            D[file.split(".")[0]] = pd.read_csv(os.path.join(root,file))
            print(file.split(".")[0])
        elif file.endswith(".xlsx"):
            D[file.split(".")[0]] = pd.read_excel(os.path.join(root, file), sheet_name=None,engine='openpyxl')
            print(file.split(".")[0])

# Sheffield
sheffield_fields = ['2024_01_MHC visit samples', '2024_01_MHC ISWT', '2024_01_MHC RHC', '2024_01_MHC IDs','MHC_Recruits Jun_27_2022','2024_05_MHC ntbnp']

# demoggraphics
dfs_demo = D[sheffield_fields[0]].copy()
vars_to_keep = {"roottable_trialnumber_value":"patient",'roottable_ethnicorigin_item':"ethnicity",
                'visit_heightn_value':"height", 'visit_weightn_value':"weight"}
dfs_demo.rename(columns=vars_to_keep,inplace=True)
dfs_demo = dfs_demo.loc[:,vars_to_keep.values()]
dfs_demo["bmi"] = dfs_demo["weight"]/dfs_demo["height"]**2
sheffield_ethnicity_dict = {'1. White British':"White British", "nan":"Unknown", '.   Asian Other':"Asian Other",
                              '.   White European':"White Other",'3. White Other':"White Other",
                                '10. Other':"Other", '7. Black Caribbean':"Black Caribbean"}
dfs_demo["ethnicity"] = dfs_demo["ethnicity"].astype(str).map(sheffield_ethnicity_dict)
dfs_demo.drop(columns = ["height","weight"],inplace = True)
dfs_demo = dfs_demo.dropna(subset=["patient"]).reset_index(drop=True)
dfs_demo["patient"] = dfs_demo["patient"].apply(lambda x:"SPVDU"+"0"*(5-len(str(int(x))))+str(int(x)))

dfs_demo0 = D[sheffield_fields[4]].copy()
dfs_demo0 = dfs_demo0.loc[:,["dob","MHC_ID"]].dropna(subset=["MHC_ID"]).reset_index(drop=True)
dfs_demo0['age'] = pd.to_datetime(dfs_demo0["dob"]).apply(lambda x:np.round((datetime.today()-x).days/365,0))
vars_to_keep = {"MHC_ID":"patient"}
dfs_demo0.rename(columns=vars_to_keep,inplace=True)
dfs_demo0 = dfs_demo0.dropna(subset=["patient"]).reset_index(drop=True)
dfs_demo0 = dfs_demo0.query("patient!='withdrawn'").reset_index(drop=True)
dfs_demo = dfs_demo.merge(dfs_demo0.loc[:,["patient","age"]],on=["patient"],how="outer")
# dfs_demo["bmi"].isna().sum()
g = dfs_demo.groupby(["patient","ethnicity"])
aux0 = g["bmi"].mean().reset_index()
aux1 = g["age"].max().reset_index()
dfs_demo = aux1.dropna(subset=["age"]).merge(aux0.dropna(subset=["bmi"]),on=["patient","ethnicity"],how="outer")

# consent dates
dfs_consent = D[sheffield_fields[3]].copy()
vars_to_keep = {"roottable_trialnumber_value":"patient",'roottable_dateofconsent_date':"date_of_consent",
                'roottable_myheartcounts_id_text':"consent_withdrawn"}
dfs_consent.rename(columns=vars_to_keep,inplace=True)
dfs_consent = dfs_consent.loc[:,vars_to_keep.values()]
dfs_consent = dfs_consent.dropna(subset=["patient"])
dfs_consent["patient"] = dfs_consent["patient"].apply(lambda x:"SPVDU"+"0"*(5-len(str(int(x))))+str(int(x)))
dfs_consent["consent_withdrawn"] = dfs_consent["consent_withdrawn"] == "withdrawn"
dfs_consent = dfs_consent.dropna()
dfs_consent.loc[:,"date_of_consent"] = pd.to_datetime(dfs_consent.loc[:,"date_of_consent"],format="%d/%m/%Y %H:%M").dt.strftime("%Y-%m-%d")
# dfs_consent["filename"] = sheffield_fields[3]

# RHC
dfs_rhc = D[sheffield_fields[2]].copy()
vars_to_keep = {'roottable_myheartcounts_truefalse':"inMHC", 'roottable_demographics_deceased':"dead",'roottable_demographics_gender':"gender","roottable_trialnumber_value":"patient",'cardiaccatheterdetails_dateofcardiaccatheter_date':"test_date"}
vars_to_keep.update({k:k.split("_")[1] for k in dfs_rhc.iloc[:,6:].keys()})
dfs_rhc.rename(columns=vars_to_keep,inplace=True)
dfs_rhc = dfs_rhc.dropna(subset=["patient"])
dfs_rhc["patient"] = dfs_rhc["patient"].apply(lambda x:"SPVDU"+"0"*(5-len(str(int(x))))+str(int(x)))
to_drop = ['roottable_myheartcounts_id_text']
dfs_rhc = dfs_rhc.drop(columns=to_drop)
cols_to_melt = dfs_rhc.iloc[:,5:].columns
dfs_rhc = dfs_rhc.melt(id_vars=["patient","inMHC","dead",'gender', 'test_date'],value_vars=cols_to_melt,var_name="variable",value_name="value")
dfs_rhc.loc[:,"test_date"] = pd.to_datetime(dfs_rhc.loc[:,"test_date"],format="%Y-%m-%d %H:%M:%S").dt.strftime("%Y-%m-%d")
dfs_rhc["filename"] = sheffield_fields[2]
# get date_diagnosis proxy
dfs_diag = dfs_rhc.loc[:,['patient','test_date']].copy().rename(columns={'test_date':'date_diagnosis'}).dropna(subset=['date_diagnosis']).sort_values(by=['patient','date_diagnosis']).drop_duplicates(subset=['patient'])
dfs_rhc = dfs_rhc.merge(dfs_diag,on='patient',how='left')

# # visit samples
# dfs_samples = D[sheffield_fields[0]].copy()

# ISWT
dfs_iswt = D[sheffield_fields[1]].copy()
vars_to_keep = {'roottable_myheartcounts_truefalse':"inMHC", 'roottable_demographics_deceased':"dead","roottable_trialnumber_value":"patient","visit_shuttledate_date":"test_date"}
vars_to_keep.update({k:"_".join(k.split("_")[:-1]) for k in dfs_iswt.iloc[:,3:].keys() if k != "visit_shuttledate_date"})
dfs_iswt.rename(columns=vars_to_keep,inplace=True)
dfs_iswt = dfs_iswt.dropna(subset=["patient"])
dfs_iswt["patient"] = dfs_iswt["patient"].apply(lambda x:"SPVDU"+"0"*(5-len(str(int(x))))+str(int(x)))
cols_to_melt = dfs_iswt.iloc[:,3:].columns
dfs_iswt = dfs_iswt.melt(id_vars=["patient","inMHC","dead",'test_date'],value_vars=cols_to_melt,var_name="variable",value_name="value")
dfs_iswt["test_date"] = pd.to_datetime(dfs_iswt["test_date"],format='mixed').dt.strftime("%Y-%m-%d")
dfs_iswt["filename"] = sheffield_fields[1]

# BNP
rename_dict = {'roottable_myheartcounts_id_text':'patient', 
 'roottable_myheartcounts_truefalse':'inMHC', 
 'roottable_demographics_deceased':'dead', 
 'labresults_date_date':'test_date','labresults_resultcode_text':'variable',
     'labresults_resultvalue_value':'value'}

dfs_bnp = D[sheffield_fields[5]].copy()
dfs_bnp = dfs_bnp.rename(columns=rename_dict)
dfs_bnp["filename"] = sheffield_fields[5]
dfs_bnp = dfs_bnp.loc[dfs_bnp.loc[:,'patient'].str.startswith('SPVDU')==True,:]# filter withdrawn
dfs_bnp["test_date"] = pd.to_datetime(dfs_bnp["test_date"],format='mixed').dt.strftime("%Y-%m-%d")
dfs_tests = pd.concat([dfs_iswt,dfs_bnp],axis=0,ignore_index=True)
dfs_tests = dfs_tests.drop_duplicates()

# merge
dfsheffield = dfs_consent.merge(dfs_demo,on="patient",how="outer",indicator=False)
dfsheffield = dfsheffield.merge(dfs_rhc,on="patient",how="outer",indicator=False)

# both          1806
# rhc_only       154
# consent_only     9x
# dfs_iswt.merge(dfs_rhc,on="patient",how="outer",indicator=True)["_merge"].value_counts()
# both          63196
# rhc_only       1484
# iswt_only         0
cols_left = ['patient', 'variable', 'value', 'filename','test_date']
cols_right = ['patient', 'date_of_consent', 'consent_withdrawn', 'inMHC', 'dead','gender', 'date_diagnosis']
dfs_tests = dfs_tests.loc[:,cols_left].merge(dfsheffield.loc[:,cols_right],on="patient",how="left",indicator=False)
# concatenate
dfsheffield = pd.concat([dfsheffield,dfs_tests],axis=0,ignore_index=True)

# add variables
dfsheffield["hospital"] = "Sheffield"
dfsheffield["date_loaded"] = datetime.now().strftime("%Y-%m-%d")

# dropnas
# dfsheffield = dfsheffield.dropna(subset=["value"]).reset_index(drop=True)

# separate values
floats = dfsheffield["value"].apply(lambda x:isinstance(x,float))
ints = dfsheffield["value"].apply(lambda x:isinstance(x,int))
strs = dfsheffield["value"].apply(lambda x:isinstance(x,str))
dates = dfsheffield["value"].apply(lambda x: isinstance(x, datetime))
aux0 = dfsheffield.loc[ints|floats,:]
aux = dfsheffield.loc[strs,:]
aux["value_cat"] = aux["value"]
aux.drop(columns=["value"],inplace=True)
aux1 = dfsheffield.loc[dates,:]
aux1["value_date"] = aux1["value"]
aux1.drop(columns=["value"],inplace=True)

dfsheffield = pd.concat([aux0,aux,aux1],axis=0,ignore_index=True)

# reformat and upoload
dfsheffield["value"] = dfsheffield["value"].astype(str)
dfsheffield["inMHC"] = dfsheffield["inMHC"].map({0:"No",1:"Yes"}).astype(str)
dfsheffield["dead"] = dfsheffield["dead"].map({0:"No",1:"Yes"}).astype(str)
dfsheffield.loc[:,[k for k in cols_final if k not in dfsheffield.keys()]] = None

# dates = ["date_of_consent","date_diagnosis","test_date","value_date","date_loaded"]
# for date in dates:
#     dfsheffield.loc[:,date] = pd.to_datetime(dfsheffield.loc[:,date])
# dfsheffield.loc[:,"date_diagnosis"] = pd.to_datetime(dfsheffield.loc[:,"date_diagnosis"])
# pdg.to_gbq(dfsheffield,"MHC_PH.clinical", project_id="imperial-410612",if_exists='replace')

# Imperial
imperial_fields = ['BASE', 'Catheter_FU', 'SixMinuteWalk-CPET-FU', 'LF-FU', 'WHO-CONSULTATIONS', 'QoL', 'VITAL SIGNS','BLOOD-FU']

# Base - demographics
df_base = D['Watch_Study_TRIPHIC_Code_2024'][imperial_fields[0]].copy()
df_match_id = D['match_imperial']
df_base = df_match_id.merge(df_base,on='OpenClinicaCode',how='right').sort_values(by='MHC_ID')

names = {'MHC_ID':"patient",'TriphicCode':'TriphicCode','Gender':"gender", 'Age':"age", 
        'BMI':"bmi", 'Ethnicity':'ethnicity','DateOfDeath':"dead",
        'DateOfMainDiagnosis':"date_diagnosis"}
df_base.rename(columns=names,inplace=True)
imperial_ethnicity_dict = {'A: British':"White British", "nan":"Unknown", 'X: Not Stated':"Unknown",
                           'H: Indian':"Asian Other",'C: Any other White Background':"White Other",
                                'S: Any other Ethnic group':"Other"}

df_base["ethnicity"] = df_base["ethnicity"].map(imperial_ethnicity_dict)
comorbidities = ['patient','With_Comorbid(Yes/No)',
                'Systemic_Hypertension', 'Diabetes', 'Diabetes_Type', 'Ischaemic_Heart',
                'Atrial_Fibrillation', 'Ischaemic_Stroke', 'COPD',
                'Parenchymal_LungDisease', 'Obesity', 'Sleep_Apnoea', 'Thyroid_Disease',
                'Past_Malignancy', 'Current_Malignancy', 'Atrial_Septal', 'Raynauds',
                'ANA_Posistive', 'Asthma']
dfcomorbidities = df_base.loc[:,comorbidities]
dfcomorbidities = dfcomorbidities.melt(id_vars=comorbidities[:2],value_vars=comorbidities[2:],var_name="comorbidity",value_name="comorbidity_value")

dfcomorbidities["comorbidity_value"].fillna("No",inplace=True)
dfcomorbidities = dfcomorbidities.query("comorbidity_value != 'No'").reset_index(drop=True)
dfcomorbidities = dfcomorbidities.groupby("patient")["comorbidity"].apply(lambda x:",".join(x)).reset_index()
# dfcomorbidities.set_index('patient', inplace=True)

variables = ['patient','RA Mean Base line', 'RVEDP Baseline', 'PA Systolic Baseline',
            'PA Diastolic Baseline', 'PA Mean Baseline', 'PCW Mean Baseline',
            'LVEDP Baseline', 'Arterial Systolic Baseline',
            'Arterial Diastolic Baseline', 'Arterial MeanBaseline', 'SAO2 Baseline',
            'SVO2 Baseline', 'Cardiac Output Measured', 'Cardiac Output', 'SFI',
            'Pulmonary Flow', 'PFI', 'QpQs', 'PVR', 'SVR', 'HeartRate',
            'ExerciseTestDate', 'ReasonOfNoWalkTest', 'WalkingDistance', 'Presats',
            'Postsats', 'DateofCMR', 'RVEDV', 'RVESV', 'RVSV', 'RVEFa', 'RVOutput',
            'LVEDV', 'LVESV', 'LVSV', 'LVEFa', 'LVOutput', 'RPAflow', 'LPAflow',
            'Aoflow', 'LungFunctionDate', 'FEV1Pred',  'FVCPred',
             'TLCPred',   'TLCOPred',
            'FIRST_TARGETED_DRUG', 'TherapyStartDate', 'TherapyEndDate',
            'SECOND_TARGETED_DRUG', 'TherapyStartDate.1', 'TherapyEndDate.1',
            'CPET_DATE', 'VO2/kg - _Max', 'PETCO2 - _Rest', 'VE/VCO2',
            'O2/HR - _Max','eGFR_Date',	'eGFR_Result']# 'TLCPredP(%)', 'TLCOPred','FEV1PredP(%)','FVCPredP(%)','TLCOPredP(%)',
dfvar = df_base.loc[:,variables]
dfvar = dfvar.melt(id_vars=variables[:1],value_vars=variables[1:],var_name="variable",value_name="value")

dfvar = dfvar.merge(dfcomorbidities,on="patient",how="outer",indicator=False)
dfvar["comorbidity"].fillna("No",inplace=True)

# add triphic code and demo
df_demo = dfvar.merge(df_base.loc[:,names.values()],on="patient",how="outer",indicator=False)
df_demo["filename"] = imperial_fields[0]

# last modifications
df_demo["gender"] = df_demo["gender"].str.capitalize()
df_demo['dead'].fillna("No",inplace=True)
ind = df_demo['dead'] != "No"
df_demo.loc[ind,'dead'] = "Yes"
df_demo = df_demo.query('variable != "WalkingDistance"').reset_index(drop=True)

# Catheter_FU - include catheter dates?
D['Watch_Study_TRIPHIC_Code_2024'][imperial_fields[1]].keys()

# SixMinuteWalk-CPET-FU
df_6min = D['Watch_Study_TRIPHIC_Code_2024'][imperial_fields[2]].copy()
df_6min.keys()
names = {'ExerciseTestDate':"test_date"}
df_6min.rename(columns=names,inplace=True)
vars = ['TriphicCode','test_date','WalkingDistance', 'Presats', 'Postsats','Ve/VCo2Gradient (L/min)', 'Vo2 (ml/min/kg)']
df_6min = df_6min.melt(id_vars=vars[:2],value_vars=vars[2:],var_name="variable",value_name="value")
# VITAL SIGNS  already included in BASE?
# D['Watch_Study_2024'][imperial_fields[6]].keys()
# add patient
df_6min = df_6min.merge(df_demo.drop(columns=[ 'variable', 'value']).drop_duplicates(),on="TriphicCode",how="inner",indicator=False)
df_6min["filename"] = imperial_fields[2]

# who consultation
df_who = D['Watch_Study_TRIPHIC_Code_2024'][imperial_fields[4]].copy()
names = {'ConsultationDate':"test_date"}
df_who.rename(columns=names,inplace=True)
vars = ['TriphicCode','test_date','WHOClass']
df_who = df_who.melt(id_vars=vars[:2],value_vars=vars[2:],var_name="variable",value_name="value_cat")

df_who = df_who.merge(df_demo.drop(columns=[ 'variable', 'value']).drop_duplicates(),on="TriphicCode",how="inner",indicator=False)
df_who["filename"] = imperial_fields[4]
df_who["variable"] = 'visit_whofunctionalclass'

df_who['value_cat'] = df_who['value_cat'].map({'3: WHO CLASS 3':'Class III', '1: WHO CLASS 1':'Class I', '2: WHO CLASS 2':'Class II','4: WHO CLASS 4':'Class IV'})
df_who.dropna(subset=["value_cat"],inplace=True)
	
# merge
dfimperial = pd.concat([df_demo,df_6min,df_who],axis=0,ignore_index=True)

####################### risk factors
# calculate risk factors
whomap = {'Class I':1,'Class II':1,'Class III':3,'Class IV':4}
whofc = dfsheffield.query("variable == 'visit_whofunctionalclass'").loc[:,['patient','value_cat']].dropna(subset=["value_cat"]).rename(columns={"value_cat":"who"})
whofc2 = dfsheffield.query("variable == 'visit_visitdate'").loc[:,['patient','value_cat']].dropna(subset=["value_cat"]).rename(columns={"value_cat":"test_date"})
whofc2['test_date'] = pd.to_datetime(whofc2['test_date'],format='mixed').dt.strftime("%Y-w%U")
whofc = whofc.merge(whofc2,on="patient",how="outer").drop_duplicates()
whofc['who'] = whofc['who'].map(whomap)
whofc['hospital'] = "Sheffield"

# calculate risk factors
# WHO FC
whofc2 = dfimperial.query("variable == 'visit_whofunctionalclass'").loc[:,['patient','value_cat','test_date']].dropna(subset=["value_cat"]).rename(columns={"value_cat":"who"})
whofc2['who'] = whofc2['who'].map(whomap)
whofc2['test_date'] = pd.to_datetime(whofc2['test_date'],format="%d/%m/%Y").dt.strftime("%Y-w%U")
whofc2['hospital'] = "Imperial"
whofc = pd.concat([whofc,whofc2],axis=0,ignore_index=True)

# Walk tests
sixmwt = dfimperial.query("variable == 'WalkingDistance'").loc[:,['patient','value','test_date']].dropna(subset=["value"]).rename(columns={"value":"6mwt"})
sixmwt['6mwt_value'] = sixmwt['6mwt']
sixmwt['6mwt'] = pd.cut(sixmwt['6mwt'],bins=[0,164,329,440,np.inf],labels=[4,3,2,1])
sixmwt['test_date'] = pd.to_datetime(sixmwt['test_date'],format="%d/%m/%Y").dt.strftime("%Y-w%U")
sixmwt['hospital'] = "Imperial"

iswt = dfsheffield.query("variable == 'visit_num_completed_shuttles'").loc[:,['patient','value','test_date']].dropna(subset=["value"]).rename(columns={"value":"iswt"})
iswt['iswt_value'] = iswt['iswt'].astype(float)*10
iswt['iswt'] = pd.cut(iswt['iswt'].astype(float)*10,bins=[0,180,250,340,np.inf],labels=[4,3,2,1])#0-180, 190-250, 260-330, >340.
iswt['test_date'] = pd.to_datetime(iswt['test_date'],format="%Y-%m-%d").dt.strftime("%Y-w%U")
iswt['hospital'] = "Sheffield"
sixmwt = pd.concat([sixmwt,iswt],axis=0,ignore_index=True)
sixmwt['walk_test'] = sixmwt['6mwt'].fillna(sixmwt['iswt'])
sixmwt = sixmwt.drop_duplicates().reset_index(drop=True)

# BNP
bnp = D['Watch_Study_TRIPHIC_Code_2024'][imperial_fields[7]].copy()
# bnp = dfimperial.query("variable in ('BNP_Date','BNP_Result','PBNP_Conversion')").pivot_table(index='patient',columns='variable',values='value',aggfunc='first').reset_index()
bnp = df_base.loc[:,['patient','TriphicCode','OpenClinicaCode','bmi']].merge(bnp,on='TriphicCode',how='right').merge(df_match_id,on='OpenClinicaCode',how='left').drop(columns=['MHC_ID','TriphicCode', 'OpenClinicaCode','eGFR_Date'])
bnp = bnp.rename(columns={"BNP_RESULT":"bnp_value","BNP_DATE":"test_date"})
bnp['bnp_value'] = (bnp['bnp_value'].astype(str).apply(lambda x:re.sub(r"[^0-9.]", "", x))).astype(float)
bnp['pbnp_value'] = np.exp(1.21+1.03*np.log(bnp['bnp_value'])-0.009*bnp['bmi']-0.007*bnp['eGFR_Result'])
bnp['test_date'] = pd.to_datetime(bnp['test_date'],format="%d/%m/%Y").dt.strftime("%Y-w%U")
# bnp['bnp'] = pd.cut(bnp['bnp'],bins=[0,50,199,800,np.inf],labels=[1,2,3,4])
bnp['pbnp'] = pd.cut(bnp['bnp_value'],bins=[0,300,650,1100,np.inf],labels=[1,2,3,4])
bnp['hospital'] = "Imperial"
# , 'BNP_Date', 'BNP_Result', 'PBNP_Conversion'
bnp3 = dfsheffield.query("variable == 'PBNP'").loc[:,['patient','value','test_date']].dropna(subset=["value"]).rename(columns={"value":"pbnp"}).drop_duplicates()
bnp3 = bnp3.rename(columns={"pbnp":"pbnp_value"})
bnp3['pbnp'] = pd.cut(bnp3['pbnp_value'].astype(float),bins=[0,300,650,1100,np.inf],labels=[1,2,3,4])
bnp3['test_date'] = pd.to_datetime(bnp3['test_date'],format="%Y-%m-%d").dt.strftime("%Y-w%U")
bnp3['hospital'] = "Sheffield"
bnp = pd.concat([bnp,bnp3],axis=0,ignore_index=True)

# transform into week year
risk = whofc.merge(sixmwt,on=["patient",'test_date','hospital'],how="outer").merge(bnp,on=["patient",'test_date','hospital'],how="outer").drop_duplicates(subset=["patient",'test_date','hospital'])
risk = risk.loc[:,['patient','hospital','test_date','who','walk_test','pbnp','6mwt','iswt','6mwt_value', 'iswt_value',  'pbnp_value','bnp_value']]
risk.loc[:,'who':] = risk.loc[:,'who':].replace('nan',np.nan).astype(float)
risk.loc[:,'risk_count'] = risk.loc[:,'who':'pbnp'].notna().sum(axis=1)
risk.loc[:,'risk_score'] = np.ceil(risk.loc[:,'who':'pbnp'].sum(axis=1)/risk.loc[:,'risk_count'])
risk["date_loaded"] = datetime.now().strftime("%Y-%m-%d")

def weekyear(x):
    try:
        datenow = x.split('-w')
        year = int(datenow[0])
        week = int(datenow[1])
        return date.fromisocalendar(year, week, 1)
    except: 
        return np.nan

risk['test_date'] = risk['test_date'].apply(weekyear)

# add group and diagnosis date
risk = risk.merge(dfm.loc[:,['patient','Group','diagnosis_date']],on='patient',how='left')
risk = risk.astype(str)
# risk = pd.concat([risk,risk2],axis=0,ignore_index=True).merge(dfm.loc[:,['patient','Group','diagnosis_date']],on='patient',how='left')
pdg.to_gbq(risk,"MHC_PH.risk2", project_id=project_id,if_exists='replace')

# add fields
dfimperial["hospital"] = "Imperial"
dfimperial["date_loaded"] = datetime.now().strftime("%Y-%m-%d")

# dropnas
# dfimperial = dfimperial.dropna(subset=["value"]).reset_index(drop=True)

# separate values
floats = dfimperial["value"].apply(lambda x:isinstance(x,float))
ints = dfimperial["value"].apply(lambda x:isinstance(x,int))
strs = dfimperial["value"].apply(lambda x:isinstance(x,str))
dates = dfimperial["value"].apply(lambda x: isinstance(x, datetime))
aux0 = dfimperial.loc[ints|floats,:]
aux = dfimperial.loc[strs,:]
aux["value_cat"] = aux["value"]
aux.drop(columns=["value"],inplace=True)
aux1 = dfimperial.loc[dates,:]
aux1["value_date"] = aux1["value"]
aux1.drop(columns=["value"],inplace=True)

dfimperial = pd.concat([aux0,aux,aux1],axis=0,ignore_index=True)

# reformat and upload
dfimperial["value"] = dfimperial["value"].astype(str)
dfimperial["inMHC"] = dfimperial["patient"].fillna("No")
ind = dfimperial["inMHC"] != "No"
dfimperial.loc[ind,"inMHC"] = "Yes"
dfimperial.loc[:,[k for k in cols_final if k not in dfimperial.keys()]] = None
for k in dfimperial.keys():
    print(k,dfimperial[k].dtype,dfsheffield[k].dtype)
    if dfimperial[k].dtype != dfsheffield[k].dtype:
        dfimperial[k] = dfimperial[k].astype(dfsheffield[k].dtype)

# merge both hospitals
df_total = pd.concat([dfsheffield,dfimperial],axis=0,ignore_index=True)

df_total["date_diagnosis"] = pd.to_datetime(df_total["date_diagnosis"]).dt.strftime("%Y-%m-%d")
df_total["date_diagnosis"] = df_total["date_diagnosis"].fillna('None')
df_total["value_date"] = pd.to_datetime(df_total["value_date"]).dt.strftime("%Y-%m-%d")
df_total["value_date"] = df_total["value_date"].str.replace("nan",'None')
df_total["test_date"] = pd.to_datetime(df_total["test_date"]).dt.strftime("%Y-%m-%d")
df_total = df_total.astype(str)
df_total["date_diagnosis"] = pd.to_datetime(df_total["date_diagnosis"],yearfirst=True,errors='coerce')
df_total["ethnicity"] = df_total["ethnicity"].fillna("Unknown",inplace=False)
ind = df_total["ethnicity"].apply(lambda x : x in ["nan","None"]) 
df_total.loc[ind,"ethnicity"] = 'Unknown'
ind = df_total["dead"].apply(lambda x : x in ["nan","None"]) 
df_total.loc[ind,"dead"] = 'No'
ind = df_total["gender"].apply(lambda x : x in ["nan","None"]) 
df_total.loc[ind,"gender"] = 'Unknown'

# map variables between them

clinical_variable_mapping = {'arterial_systolic':
                        [ 'visit_systolic_pre_1','arterialsystolic', 
                        'visit_systolic_pre_2', 'visit_systolic_post_1',
                            'visit_systolic_post_0', 'visit_systolic_post_2',
                            'Arterial Systolic Baseline','PA Systolic Baseline', 'pasystolic'], 
                        'arterial_diastolic':
                            ['visit_diastolic_pre_0', 'visit_diastolic_pre_1',
                            'visit_diastolic_pre_2', 'visit_diastolic_post_0',
                            'visit_diastolic_post_1', 'visit_diastolic_post_2',
                            'Arterial Diastolic Baseline','PA Diastolic Baseline',
                                'padiastolic',  'arterialdiastolic'],

                        'arterial_mean':
                            ['PA Mean Baseline',
                            'pamean',
                            'Arterial MeanBaseline'],

                        'cardiac_output':
                        [ 'cardiacoutput', 
                        'Cardiac Output Measured',
                        'Cardiac Output'],
                        'pcw':
                        ['pcw',
                        'PCW Mean Baseline'], 

                        'bnp':
                        ['BNP_Date', 'BNP_Result'],

                        'heart_rate':
                        ['visit_start_hr',
                            'visit_highest_hr', 'visit_postrest_hr', 'visit_hr_pre_0', 'visit_hr_pre_1', 'visit_hr_pre_2',
                            'visit_hr_post_0', 'visit_hr_post_1', 'visit_hr_post_2',
                            'visit_systolic_pre_mean', 'visit_diastolic_pre_mean',
                            'visit_hr_pre_mean', 
                            'HeartRate','heartrate', 
                            ], 

                        'PVR':['pvr', 'fickpvr', 
                            'PVR',
                            'FEV1Pred', 'FEV1PredP(%)', 'FVCPred', 'FVCPredP(%)'],

                        'VO2':['sao21','O2/HR - _Max',
                            'VO2/kg - _Max',
                            'visit_onoxygen',
                            'visit_o2rate',
                            'VE/VCO2', 
                            'Ve/VCo2Gradient (L/min)',
                                'Vo2 (ml/min/kg)', 
                                'SAO2 Baseline', 'SVO2 Baseline',
                                'visit_start_sao2', 'visit_lowest_sao2'],

                        'walking_distance':
                                ['WalkingDistance', 
                                'visit_distance'],
                        'RV':['RVEDV', 'RVESV', 'RVSV', 'RVEFa', 'RVOutput', 'rvsystolic', 'RVEDP Baseline'],

                        'other':
                                ['Presats', 'Postsats',
                            
                            'LVEDV', 'LVESV',
                            'LVSV', 'LVEFa', 'LVOutput', 'RPAflow', 'LPAflow', 'Aoflow',
                                'TLCPred','SVR',
                            'TLCPredP(%)', 'TLCOPred', 'TLCOPredP(%)', 
                            'PETCO2 - _Rest', 'CPET_DATE',
                            'visit_visitnumber', 'visit_whofunctionalclass',
                            'visit_borg_pre',       'visit_borg_post', 'visit_shuttle_comment', 
                            'ramean', 
                            'ReasonOfNoWalkTest', 
                            'FIRST_TARGETED_DRUG', 'SECOND_TARGETED_DRUG',
                            'ExerciseTestDate',
                            'DateofCMR', 'LungFunctionDate', 'TherapyStartDate',
                            'TherapyEndDate', 'TherapyStartDate.1', 'TherapyEndDate.1','visit_num_completed_shuttles',
                            'visit_shuttle_term_reason', 'visit_visitdate',
                            'visit_shuttledate', 
                            'RA Mean Base line',    
                            'LVEDP Baseline',
                            'SFI', 'Pulmonary Flow', 'PFI', 'QpQs','cardiacindex']}
clinical_variable_mapping_inv = {c:k for k in clinical_variable_mapping.keys() for c in clinical_variable_mapping[k]}
        
df_total['variable_group'] = df_total['variable'].map(clinical_variable_mapping_inv)

# remove unnecessary nans
idx = df_total.loc[:,'value_date']=='nan'
idx2 = df_total.loc[:,'value']=='nan'
idx3 = df_total.loc[:,'value_cat']=='nan'
idx = idx & idx2 & idx3
df_total = df_total.drop(index=np.where(idx)[0]).reset_index(drop=True)

# deal with dates
idx = df_total.loc[:,'value_date']!='nan'
df_total.loc[idx,'test_date'] = df_total.loc[idx,'value_date']
df_total.drop(columns=['value_date'],inplace=True)

# add group and sink
df_total = df_total.merge(dfm.loc[:,['patient','Group']],on='patient',how='left')
pdg.to_gbq(df_total,"MHC_PH.clinical2", project_id="imperial-410612",if_exists='replace')
