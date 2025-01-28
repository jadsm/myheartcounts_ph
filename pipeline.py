import os
from utils.constants import *
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path 
from utils.utils_ml import *
from utils.utils_pipeline import *

suffix = "_us2"

# load the data
df2,df3,df4,df4_std,df5 = reload_and_save_data(suffix,cohort='UK',reload_data=False)
df2_us,_,_,_,_ = reload_and_save_data(suffix,cohort='US',reload_data=False)
df2['cohort'] = 'UK'
df2_us['cohort'] = 'US'

# correction of single patient ethnicity
idx = df2_us['patient'] == 'DwqpYs6vVj-4RMnhHrYPBKjy'
df2_us.loc[idx,'ethnicity'] = 'Mixed'

# get all the demographics data
cols=['cohort','patient','Group','age', 'gender','ethnicity','bmi']
aa = pd.concat([df2,df2_us],axis=0,ignore_index=True).loc[:,cols].drop_duplicates()
aa['bmi'] = pd.cut(aa['bmi'], bins=[0, 20, 25, 30, np.inf],labels=['Underweight','Normal','Overweight','Obese']).astype(str).str.replace('nan','Unknown')
aa['age'] = pd.cut(aa['age'], bins=[0, 30, 50, 70, np.inf],labels=['<30','30-50','50-70','>70']).astype(str).str.replace('nan','Unknown')
aa = aa.fillna('Unknown')
g = aa.groupby(['cohort','Group'])
pd.concat([g[ii].value_counts() for ii in cols[3:]],axis=0).to_csv('data/demographics.csv')

# binarise ethnicity
df2['ethnicity'] = df2['ethnicity'].map({'White British':'White','White Other':'White'}).fillna('Other')
df2_us['ethnicity'] = df2_us['ethnicity'].map({'White British':'White','White Other':'White'}).fillna('Other')

# prediagnosis analysis - decline
slopes = analyse_prediagnosis(df2,suffix=suffix, prediagnosis_analysis=False, plot_flag=False,overlapping_windows=False)

################# Plot the raw data ###################
plot_raw_data(df2,df3,df4,df5,suffix=suffix,plot_flag=False)
 
######### data preparation for walk tests
compute_walktest(agg_lvl='quarter',aggregation='median',suffix=suffix,flag_6mwt=False,reload_data=False)

################# Frequentist analysis - between groups for each variable ###################
statistical_analysis(df2,slopes,stat_analysis=False,freq_analysis=False,plot_flag=False)

################# cohort comparison ################
cohort_comparison(df2,df2_us,plot_flag=False)

################ Feature Extraction ###################
dfq,df_features_all = extract_features(df2,extract_feat_flag=False)
dfq_us,df_features_all_us = extract_features(df2_us,cohort='US',extract_feat_flag=False)

#### run ML models
# run ML model by groups of activity
feature_spaces = ['activity'] 
ml_pipeline_activity_wrapper(dfq,df_features_all,dfq_us,df_features_all_us,ml_pipe_flag=False,feature_groups=['activity-group-device','device'],feature_spaces = feature_spaces)

# run ML model by groups of questions
feature_spaces = ['activity_questionnaire'] 
activity_include = ['A_iPhone','A_Watch',{'device':'Watch'}]#,{'device':'iPhone'},{'device':'Watch'}
ml_pipeline_quest_wrapper(dfq,df_features_all,dfq_us,df_features_all_us,activity_include = activity_include,ml_pipe_flag=False,feature_spaces = feature_spaces)

# all features
ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='',lbl = 'all',feature_spaces = feature_spaces)

# apply the two top models to external cohort
for quest_incl,activity_include_now in [[['CARDIO DIET SURVEY','Vaping_Smoking','ACTIVITY AND SLEEP SURVEY'],'A_iPhone'],
                                        [['SATISFIED SURVEY'],{'device':'Watch'}],
                                        [[''],'A_Watch']]:#[['SATISFIED SURVEY'],'A'],
    # select the features
    df_features_allnow,df_features_all_usnow,lbl = activity_selector(activity_include_now,df_features_all,df_features_all_us)
    dfqnow,dfq_usnow = question_selector(quest_incl,dfq,dfq_us)
    feature_spaces = ['activity_questionnaire'] if len(quest_incl[0]) > 0 else ['activity']
    # run the pipeline for external
    ml_pipeline([dfqnow,df_features_allnow],[dfq_usnow,df_features_all_usnow],ml_pipe_flag=True,external_flag='_external',lbl = lbl+'_'.join(quest_incl),feature_spaces = feature_spaces)
            
    # Data drift
    calc_data_drift([dfqnow,df_features_allnow],[dfq_usnow,df_features_all_usnow])

    # fine tuning
    ml_pipeline([dfqnow,df_features_allnow],[dfq_usnow,df_features_all_usnow],ml_pipe_flag=True,external_flag='_finetune',finetune_test_size=.2,lbl = lbl+'_'.join(quest_incl),feature_spaces = feature_spaces)

    print('Finished',quest_incl,activity_include_now)

# this is the older version of the pipeline
# # filter irrelevant questions
# cols = [c for c in exclusions_questionnaire if c in dfq.keys()]
# dfq.drop(columns=cols,inplace=True)
# cols = [c for c in exclusions_questionnaire if c in dfq_us.keys()]
# dfq_us.drop(columns=cols,inplace=True)

# # repeat for all features plus heart condition
# ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='',lbl = 'all_HC')
# ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='_external',lbl = 'all_HC')

# # repeat for all features
# dfq.drop(columns=['heartCondition'],inplace=True)
# ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='',lbl = 'all')
# ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='_external',lbl = 'all')

# # Data drift
# calc_data_drift([dfq,df_features_all],[dfq_us,df_features_all_us])

# # fine tuning
# ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='_finetune',finetune_test_size=.2,lbl = 'all')

# # for finetune_test_size in [.1,.3,.4,.5]:
# #     ml_pipeline([dfq,df_features_all],[dfq_us,df_features_all_us],ml_pipe_flag=True,external_flag='_finetune',finetune_test_size=finetune_test_size,quest_lbl = 'all')