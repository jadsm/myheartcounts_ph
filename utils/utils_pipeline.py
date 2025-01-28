import pandas as pd
import pandas_gbq as pdg
import numpy as np
from utils.constants import *
import altair as alt
import statsmodels.formula.api as smf
from utils.utils_altair import *
from utils.utils_ml import *
from utils.utils_timeseries import stat_feature_extraction,fft_feature_extraction,ARIMA_feature_extraction,CC_feature_extraction
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import shap
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_validate
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from xgboost import plot_tree
from sklearn.svm import SVC

def reload_and_save_data(suffix,cohort='UK',reload_data=True):
    if reload_data:
        df = pdg.read_gbq(f"""select DATE_TRUNC(cast(startTime as date), MONTH) startTime, device,`Group`,consent_date,diagnosis_date,patient,variable,
                        avg(value) value, hospital,Withdrawn,ethnicity,dead
                        from `MHC_PH.activity{suffix}`
                        where clean_status='clean_1' and cohort = '{cohort}' and device_rank in ('Watch1','iPhone1')
                        group by DATE_TRUNC(cast(startTime as date), MONTH),`Group`,device,patient,variable,consent_date,diagnosis_date,hospital,Withdrawn,ethnicity,dead""",project_id=project_id)

        # df['isUK'] = df['patient'].str.startswith('SPVDU')
        # df['SuperGroup'] = df['Group'].map({'Not found':'Not found', 'Healthy':'Healthy', 'PAH':'PH', 'COVID':'DC', 'NotPH':'DC', 'PH5':'PH'})
        # df.groupby(['isUK','SuperGroup','device']).agg({'patient':['nunique','count'],'variable':'nunique','startTime':['min','max']}).reset_index().to_csv('data/UK_US_summary_data.csv',index=False)

        # ################ data preparation
        g = df.groupby(['variable','patient','Group'])
        aux = ((g["startTime"].max() - g["startTime"].min()).dt.days//30).reset_index()
        aux.rename(columns={"startTime":"duration_months"},inplace=True)
        aux["duration_months_groups"] = pd.cut(aux["duration_months"], [0,6,12,24,36,np.inf], labels=["<6 months",">6 months",">1 years",">2 years",">3 years"])
        aux2 = aux.groupby(['variable','duration_months_groups','Group'])["patient"].nunique().reset_index()
        pdg.to_gbq(aux2,f"MHC_PH.patient_counts{suffix}", project_id="imperial-410612",if_exists="replace")

        # remove anything below 3 years - for stepcount / 1 year for the rest - remove Height / Weight
        # patients_stepcount = aux.query("variable == 'StepCount' and duration_months_groups == '>3 years'")
        # patients_rest = aux.query("variable not in ('StepCount','Height','Weight','bmi') and duration_months_groups in ('>1 years','>2 years','>3 years')")
        # patients_rest2 = aux.query("variable == 'bmi'")
        # aux3 = pd.concat([patients_stepcount,patients_rest,patients_rest2],axis=0).reset_index(drop=True)
        # df = df.merge(aux3.loc[:,["variable","patient"]],on = ["variable","patient"],how='inner')

        # stitch together - to 3 years + / 1 year +
        df['min_time'] = df.groupby(['Group','patient','variable','device'])["startTime"].transform('min')
        df.loc[:,"days"] = (df.loc[:,"startTime"] - df.loc[:,"min_time"]).dt.days
        df.loc[:,"months"] = df.loc[:,"days"]//30

        # calculate the other months
        df['months_diagnosis'] = (pd.to_datetime(df['startTime'],utc=True) - pd.to_datetime(df['diagnosis_date'],utc=True)).dt.days//30
        df['months_consent'] = (pd.to_datetime(df['startTime'],utc=True) - pd.to_datetime(df['consent_date'],utc=True)).dt.days//30

        # curate sleep data - remove 'inbed' if they do not have either awake or asleep
        # idvars = [k for k in df.keys() if k not in ['value','variable']]
        # aux = df.query('variable in ("Awake","Asleep","InBed")').pivot_table(index=idvars,
        #     columns='variable',values='value').reset_index()
        # aux = pd.melt(aux.dropna(subset=['Awake','Asleep'],how='all'),id_vars=idvars,value_vars=['Awake','Asleep','InBed'])
        # df = pd.concat([df.query('variable not in ("Awake","Asleep","InBed")'),
        #                 aux],axis=0,ignore_index=True)

        # filter and compute aggregations
        cols_to_keep = ["Group","patient",'device',"variable",'months','value','months_diagnosis','months_consent','hospital', 'Withdrawn', 'ethnicity', 'dead']
        df2 = df.loc[:,cols_to_keep].copy()

        df2["Group"] = df2["Group"].map(group_dict)
        # if date of diagnosis or consent are missing, use the date of beginning of time
        df2['months_diagnosis'] = df2['months_diagnosis'].fillna(df2['months'])
        df2['months_consent'] = df2['months_consent'].fillna(df2['months'])

        # remove bedbound if it is less than 0.1
        df2 = df2.drop(index=df2.query('variable == "BedBound" and value < 0.1').index).reset_index(drop=True)

        df3 = df2.groupby(["Group","variable",'device','months'])["value"].mean().reset_index()
        # df4 = df2.groupby(["Group","variable",'device','months_diagnosis'])["value"].mean().reset_index().rename(columns={'months_diagnosis':'months'})
        df4 = df2.groupby(["Group","variable",'device','months_diagnosis']).agg({'value':'mean','patient':'nunique'}).reset_index().rename(columns={'months_diagnosis':'months','patient':'n_patient'})
        df5 = df2.groupby(["Group","variable",'device','months_consent'])["value"].mean().reset_index().rename(columns={'months_consent':'months'})

        # now standardize the data for slope calculations
        df2['mean_by_patient'] = df2.groupby(["patient","variable",'device'])['value'].transform(np.mean)
        df2['value_std'] = df2['value'] - df2['mean_by_patient']
        df4_std = df2.groupby(["Group","variable",'device','months_diagnosis']).agg({'value_std':'mean','patient':'nunique'}).reset_index().rename(columns={'months_diagnosis':'months','value_std':'value','patient':'n_patient'})

        
        # # recalculate bmi - this is nolonger needed bc it has been calculated in ETL - leave here until testing complete
        # df_bmi = pdg.read_gbq(f"""select patient,variable, avg(value) value
        #                   from MHC_PH.activity{suffix}
        #                   where clean_status='raw' 
        #                   and variable in ('Height','BodyMass')
        #                     and aggregation = 'Mean'
        #                   group by patient,variable""",project_id=project_id)

        # df_bmi = df_bmi.pivot(index="patient",values="value",columns="variable")
        # df_bmi["Height_inferred"] = df_bmi["Height"].isna() 
        # df_bmi["Height"] = df_bmi["Height"].fillna(1.7,inplace=False)
        # df_bmi["bmi"] = df_bmi["BodyMass"]/(df_bmi["Height"]**2)
        # df_bmi["bmi_cat"] = pd.cut(df_bmi["bmi"], [0,18.5,24.9,29.9,np.inf], labels=["Underweight","Normal","Overweight","Obese"])
        # # print(df_bmi.loc[~df_bmi["Height_inferred"],"bmi_cat"].value_counts())
        # # print(df_bmi["bmi_cat"].value_counts())
        # df2 = df2.merge(df_bmi.loc[:,["bmi","bmi_cat"]],on="patient",how="left")

        # add bmi as column
        newcol = df2.query('variable == "bmi"').loc[:,['patient','value']].rename(columns={'value':'bmi_mhc'})
        df2 = df2.merge(newcol,on="patient",how="left")

        # add clinical data
        vars_to_keep = ['patient', 'age', 'gender',  'test_date'] # 'date_of_consent','date_diagnosis','variable_group'
        df_clinical = pdg.read_gbq(f"""select * from (select patient,{",".join(['min('+k+') '+k for k in vars_to_keep[1:]])} from MHC_PH.clinical2
                                where patient is not null or patient in ('nan','None','NaN') group by patient) T 
                                join (SELECT patient,AVG(CAST(bmi AS FLOAT64)) bmi FROM MHC_PH.clinical2
                                        where bmi != 'nan' group by patient) using(patient)
                                """,project_id=project_id)

    # supplement with age and gender and validate with clinical data

        df_ag = pdg.read_gbq("""SELECT patient,max(value) age FROM `imperial-410612.MHC_PH.questionnaire_us2` 
                            where Question in ('heartAgeDataAge','NonIdentifiableDemographics.patientCurrentAge')
                            group by patient""",project_id=project_id)
        
        df_sex = pdg.read_gbq("""SELECT patient,REPLACE(max(value),'HKBiologicalSex','') gender FROM `imperial-410612.MHC_PH.questionnaire_us2` 
                                where lower(value) like '%male%'
                                group by patient""",project_id=project_id)
        
        df_ethnicity = pdg.read_gbq("""SELECT value_clean ethnicity,patient FROM `imperial-410612.MHC_PH.questionnaire_us2`    
                                    where Question = 'race'""",project_id=project_id)

        us_ethnicity_dict = {'White':"White Other",
                        "nan":"Unknown",
                        'Some other race':'Other',
                        'White, and Black, African-American, or Negro':'Mixed',
                        'Asian Indian':"Asian Other",
                        'White, and American Indian':'Mixed',
                        'Filipino':"Asian Other",
                        'Black, African-American, or Negro':'Black',
                        'White, and Black, African-American, or Negro, and American Indian':'Mixed',
                        'Chinise':'Asian',
                        'White, and Pacific Islander':'Mixed'}
        df_ethnicity['ethnicity'] = df_ethnicity['ethnicity'].map(us_ethnicity_dict)

        # df_ag = df_ag.query(f'patient not in {tuple(df_clinical["patient"].unique())}')
        df_clinical = df_clinical.merge(df_ag,on="patient",how="outer",suffixes=['','_q']).merge(df_sex,on="patient",how="outer",suffixes=['','_q'])
        df_clinical['gender'] = df_clinical['gender'].fillna(df_clinical['gender_q'])
        df_clinical['age'] = df_clinical['age'].replace('nan',np.nan).astype(float)
        df_clinical['age'] = df_clinical['age'].fillna(df_clinical['age_q']).astype(float)
        df_clinical.drop(columns=['age_q', 'gender_q'],inplace=True)

    # condition clinical data and merge to df2
    # df2.rename(columns={"bmi":"bmi_mhc","bmi_cat":"bmi_cat_mhc"},inplace=True)
        df2 = df2.merge(df_clinical,on="patient",how="outer").merge(df_ethnicity,on="patient",how="left",suffixes=['','_q'])
        df2['ethnicity'] = df2['ethnicity'].fillna(df2['ethnicity_q'])
        df2['bmi'] = df2['bmi'].fillna(df2['bmi_mhc'])
        df2.drop(columns=['bmi_mhc','ethnicity_q'],inplace=True)

    # df2.query('bmi == bmi & variable == variable')['patient'].nunique()
    # 103 - 7 without
    # df2.query('gender == gender & variable == variable')['patient'].nunique()
    # 110 - 72 without
    # df2.query('age == age & variable == variable')['patient'].nunique()
    # 88 - 2 without
    # df2.query('ethnicity == ethnicity & variable == variable')['patient'].nunique()
    # 107 - 3 without


        df2.to_csv(f"data/temp24{suffix}{cohort}.csv",index=False)
        df3.to_csv(f"data/temp34{suffix}{cohort}.csv",index=False)
        df4.to_csv(f"data/temp44{suffix}{cohort}.csv",index=False)
        df4_std.to_csv(f"data/temp44_std{suffix}{cohort}.csv",index=False)
        df5.to_csv(f"data/temp54{suffix}{cohort}.csv",index=False)
    else:
        df2 = pd.read_csv(f"data/temp24{suffix}{cohort}.csv")
        df3 = pd.read_csv(f"data/temp34{suffix}{cohort}.csv")
        df4 = pd.read_csv(f"data/temp44{suffix}{cohort}.csv")
        df4_std = pd.read_csv(f"data/temp44_std{suffix}{cohort}.csv")
        df5 = pd.read_csv(f"data/temp54{suffix}{cohort}.csv")

    # get percentages
    aux = df2.loc[:,['patient','bmi','gender','ethnicity','hospital','Group']]
    aux['bmi_cat'] = pd.cut(aux['bmi'],bins=[0,20,25,30,35,80],labels=['underweight','normal','overweight','obese','morbid'])
    for var in ['bmi_cat','gender','ethnicity','hospital','Group']:
        aux[var] = aux[var].astype(str).fillna('missing')
        print(var,aux.groupby(var)['patient'].nunique())

    df2 = df2.query('Group in ("Healthy","DC","PH")')

    print("data loaded!")

    return df2,df3,df4,df4_std,df5 

def analyse_prediagnosis(df2,suffix='',prediagnosis_analysis=True,plot_flag=True,overlapping_windows=False):
    if prediagnosis_analysis:
        slopes = pd.DataFrame(columns=['Group','patient','variable','device','time_segment','slope','pvalue','n'])
        
        time_segments = [-60,-24,-18,-12,-9,-6,-3,3,6,9,12,18,24] if overlapping_windows else [-60,-24,-18,-12,-9,-6,-3,0,3,6,9,12,18,24]
        time_segments_ini = 0 if overlapping_windows else 1
        standardiser = df2.groupby(['variable','device'])['value'].mean().to_dict()

        df2['months'] = df2['months_diagnosis']

        for si in range(time_segments_ini,len(time_segments)):
        
            # filter data by 6 month chunks
            # aux = df4.query(f'(Group== "PH" and months>={time_segments[si-1]} and months<={time_segments[si]}) or Group!= "PH"')
            # filter data with reference to baseline
            if overlapping_windows:
                ref1 = time_segments[si] if time_segments[si]<0 else 0
                ref2 = 0 if time_segments[si]<0 else time_segments[si]
            else:
                ref1 = time_segments[si-1]
                ref2 = time_segments[si]

            aux = df2.query(f'(Group== "PH" and months>={ref1} and months<={ref2})')
            if aux.shape[0]==0:
                continue
            # linear fit of the data - overall delta?
            for gi,g in aux.groupby(['Group','patient','variable','device']):

                print(gi)

                linear_model = (g.iloc[-1,5]-g.iloc[0,5])/np.abs(g.iloc[0,4]-g.iloc[-1,4])/standardiser[gi[2:]]

                
                # slopes = pd.concat([slopes,pd.DataFrame([[gi[0],gi[1],gi[2],str(time_segments[si-1])+'to'+str(time_segments[si]),linear_model[0],V[0,0],g.shape[0]]],columns=slopes.columns)],axis=0,ignore_index=True)
                slopes = pd.concat([slopes,pd.DataFrame([[gi[0],gi[1],gi[2],gi[3],str(ref1)+'to'+str(ref2),linear_model,0,g.shape[0]]],columns=slopes.columns)],axis=0,ignore_index=True)
        

        # order by group, variable, slope
        slopes['slope_abs'] = slopes['slope'].abs()
        slopes['slope_max'] = slopes.groupby(['Group','patient','variable','device'])['slope_abs'].transform('max')
        slopes = slopes.sort_values(['Group','slope_max'],ascending=[True,False])
        slopes['unit_labels'] = slopes['variable'].map(unit_labels)
        slopes.to_csv(f'data/slopes{suffix}.csv',index=False)

        if plot_flag:
            slopes['x'] = 0
            base = alt.Chart(slopes).mark_boxplot(opacity=.7).transform_filter(alt.FieldEqualPredicate(field='Group', equal='PH'))

            slopes_chart = base.encode(y=alt.Y('variable:N',sort=list(slopes['variable'].unique())),
                                            x=alt.X('slope:Q', axis=alt.Axis(tickCount=5)),
                                            color=alt.Color('Group:N').scale(domain=list(palette.keys()),range=list(palette.values())),
                                            tooltip=['variable','device','slope','Group','time_segment'],
                                            )

            reference_line= alt.Chart(slopes).mark_rule(color='black', size=1).encode(x=alt.X('x').title('slope'))  
            #.encode(y='mean(slope):Q',size=alt.value(2)).transform_filter(alt.FieldOneOfPredicate(field='variable',

            #.facet(column='time_segment:N')
            chart = (slopes_chart+reference_line).properties(width=800,height=500).facet(column = 'device',columns=2).configure_axis(
            labelFontSize=14,
            titleFontSize=20
            ).configure_legend(disable=True)
            chart.save(f"frontend/www/slopes22{suffix}.html")

            # plot the same with temporality
            base = alt.Chart(slopes).transform_filter(alt.FieldEqualPredicate(field='Group', equal='PH'))

            reference_line2= base.mark_rule(color='black', size=1).encode(y=alt.Y('x').title('slope'))  

            slopes_chart2 = base.mark_line(opacity=.7).encode(y=alt.Y('slope:Q', axis=alt.Axis(tickCount=5)),
                                        x=alt.X('time_segment:N').title('months prior to diagnosis'),
                                        color=alt.Color('variable:N').title(None))

            slopes_points2 = base.mark_circle(opacity=.7).encode(y=alt.Y('slope:Q', axis=alt.Axis(tickCount=5)),
                                        x=alt.X('time_segment:N').title('months prior to diagnosis'),
                                        color=alt.Color('variable:N').title(None),
                                        size=alt.Size('n:Q',scale=alt.Scale(range=[50,200])),
                                        tooltip=['variable',
                                                    alt.Tooltip('slope',format=".3f"),
                                                    alt.Tooltip('time_segment',title='months prior to diagnosis'),
                                                    'n'])

            chart = (slopes_chart2+slopes_points2+reference_line2).properties(width=800,height=500).facet(column = 'device',columns=2).configure_axis(
                labelFontSize=20,
                titleFontSize=20
            ).configure_legend(labelFontSize=14,
                titleFontSize=20,
                symbolLimit=0)
            # chart.save("frontend/www/slopes32.html")

            # plot the same with temporality
            vars = ['InBed', 'FlightsClimbed', 'FlightsClimbedPaceMean','FlightsClimbedPaceMax','StepCountPaceMax','StepCountPaceMean',
                'HeartRateReserve', 'AppleStandTime', 'HeartRateVariabilitySDNN',
                'VO2Max', 'RestingHeartRate', 'HeartRate','CardiacEffort',
                'WalkingHeartRateAverage', 'BasalEnergyBurned','ActiveEnergyBurned',
                'StepCount']
            chart = alt.Chart(slopes).mark_bar(size=60,opacity=.5).encode(y=alt.Y('variable', axis=alt.Axis(labelAngle=0), sort=alt.EncodingSortField(field="slope", op="mean", order='ascending')),
                                                        x=alt.X('slope', axis=alt.Axis(tickCount=10)).stack(None),
                                                        color=alt.Color('variable'),
                                                        #  order=alt.Order("slope:N", sort='descending'),   
                                                        tooltip=['variable','Group','slope','pvalue']).transform_filter(alt.FieldOneOfPredicate(field='variable', 
                                                                                                                                                oneOf=vars)).transform_filter(alt.FieldEqualPredicate(field='time_segment',
                                                                                                                                                                                                        equal="0")).transform_filter(alt.FieldEqualPredicate(field='Group', 
                                                                                                                                                                                                                                                            equal="PH"))

            print('slopes replotted')
        print('slopes recomputed')
    else:  
        slopes = pd.read_csv(f'data/slopes{suffix}.csv')
        print('sloped loaded from disk')
    
    # perform ANOVA TEST    
    from scipy.stats import f_oneway
    res = []
    for vi,v in slopes.groupby(['variable',  'device']):
        print(vi)
        v = v.query("time_segment in ('-3to0', '0to3','3to6')")
        l = [ll[1]['slope'].dropna().values for ll in list(v.groupby('time_segment'))]
        try:
            res.append({'variable':vi[0],
                        'device':vi[1],
                        'pvalue':f_oneway(*l).pvalue})
        except Exception as e: 
            print(vi)
            print(e)
        # sm.stats.anova_lm(v, formula='slope ~ time_segment')
    dfres = pd.DataFrame(res).sort_values(by='pvalue',ascending=True)
    dfres.to_csv('data/slope_pvalues2.csv',index=False)
    return slopes


def plot_raw_data(df2,df3,df4,df5,suffix='',plot_flag=True):
    if plot_flag:    
        # this plots the average

        for device in ['Watch','iPhone']:
            # this plots each patient
            chart2 = make_plot_allvars(df2,device, by='patient',opacity=.5,color_range = palette)#,color_range=["#DFFF00","#FFBF00","#FF7F50","#DE3163","#9FE2BF","#40E0D0","#6495ED","#CCCCFF"])
            chart2.save(f"frontend/www/plot6{device}{suffix}.html")

            chart = make_plot_allvars_ci(df2,df3,device,color_range = palette)
            chart.save(f"frontend/www/plot4{device}{suffix}.html")

            df4.rename(columns={"months_diagnosis":"months"},inplace=True)
            df2now = df2.copy().drop(columns=['months']).rename(columns={"months_diagnosis":"months"},inplace=False)
            chart = make_plot_allvars_ci(df2now,df4,device,color_range = palette,xscale=[[-20,20],[-100,100]])
            chart.save(f"frontend/www/plot7{device}{suffix}.html")

            df5.rename(columns={"months_consent":"months"},inplace=True)
            df2now = df2.copy()
            df2now['months_consent'] = df2now['months_consent'].fillna(df2now['months'])
            df2now = df2now.drop(columns=['months']).rename(columns={"months_consent":"months"},inplace=False)
            
            # remove outliers for visualisation purposes
            idx = df2now.query("variable == 'StepCountPaceMax' and value > 500").index
            df2now.drop(index=idx,inplace=True)
            idx = df5.query("variable == 'StepCountPaceMax' and value > 500").index
            df5.drop(index=idx,inplace=True)
            
            chart = make_plot_allvars_ci(df2now,df5,device,color_range = palette,xscale=[[-20,20],[-100,100]])
            chart.save(f"frontend/www/plot8{device}{suffix}.html")

            # repeat these with date of diagnosis
            df2aux = df2.copy()
            df2aux['months_consent'].fillna(df2aux['months'],inplace=True)
            df2aux['months_diagnosis'].fillna(df2aux['months'],inplace=True)
            df2aux.drop(columns=['months'],inplace=True)
            df2aux.rename(columns={"months_consent":"months"},inplace=True)
            chart2 = make_plot_allvars(df2aux,device,by='patient',opacity=.5,color_range = palette)#,color_range=["#DFFF00","#FFBF00","#FF7F50","#DE3163","#9FE2BF","#40E0D0","#6495ED","#CCCCFF"])
            chart2.save(f"frontend/www/plot9{device}{suffix}.html")
            
            # repeat with date of diagnosis
            df2aux.drop(columns=['months'],inplace=True)
            df2aux.rename(columns={"months_diagnosis":"months"},inplace=True)
            chart2 = make_plot_allvars(df2aux,device,by='patient',opacity=.5,color_range = palette)#,color_range=["#DFFF00","#FFBF00","#FF7F50","#DE3163","#9FE2BF","#40E0D0","#6495ED","#CCCCFF"])
            chart2.save(f"frontend/www/plot10{device}{suffix}.html")
        print('raw data successfully re-plotted')
    else:
        print('raw data not re-plotted')


def compute_walktest(agg_lvl,aggregation,suffix,flag_6mwt,reload_data):
    if flag_6mwt:

        if reload_data:
            df_6mwt_cl = pdg.read_gbq("""SELECT patient,`Group`,hospital,
                                who who_fc, pbnp pbnp_score,walk_test walk_score,pbnp_value, `6mwt_value`, iswt_value,risk_score,
                                test_date FROM `imperial-410612.MHC_PH.risk` 
                                        """,project_id=project_id)
            df_6mwt_cl.loc[:,'who_fc':'iswt_value'] = df_6mwt_cl.loc[:,'who_fc':'iswt_value'].astype(float) 
            df_6mwt_cl = df_6mwt_cl.drop_duplicates().reset_index(drop=True)

            # PARSE_DATE('%Y-w%W',test_date)
            # df_6mwt_mhc = pdg.read_gbq(""" """,project_id=project_id)
    # and variable not in ('StepCount','StepCountPaceMax','StepCountPaceMean','FlightsClimbed','FlightsClimbedPaceMax','FlightsClimbedPaceMean')
    #                                   union all 
    #                                   select patient, value distance,startTime test_date,`Group`,variable,hospital,diagnosis_date from `imperial-410612.MHC_PH.activity_nodup` 
    #                                     where clean_status = 'imputed' and variable in ('StepCount','StepCountPaceMax','StepCountPaceMean','FlightsClimbed','FlightsClimbedPaceMax','FlightsClimbedPaceMean')
            df_act_mhc = pdg.read_gbq(f"""select patient,DATE_TRUNC(cast(test_date as date), MONTH) test_date_mhc,`Group`,variable,device,hospital,diagnosis_date, avg(distance) act from (
                                    select patient, value distance,startTime test_date,`Group`,variable,device,hospital,diagnosis_date from `imperial-410612.MHC_PH.activity{suffix}` 
                                        where clean_status = 'clean_1' and device_rank in ('Watch1','iPhone1')
                                        union all 
                                    SELECT patient,distance,startDate test_date,`Group`,"6mwt MHC" variable,'Watch' device,hospital,diagnosis_date FROM `imperial-410612.MHC_PH.6mwt` 
                                        join `imperial-410612.MHC_PH.id_mapping2` using(patient)
                                )       T
                                group by patient,DATE_TRUNC(cast(test_date as date), MONTH),`Group`,variable,device,hospital,diagnosis_date """,project_id=project_id)
            
            # aa = [patient,test_date_mhc,atwork,beneficial,body_remarkable_self_healing,body_self_healing_from_most_conditions_and_diseases,body_self_healing_in_many_different_circumstances,chestPain,chestPainInLastMonth,convenient,currentSmokeless,currentSmoking,currentVaping,disease,dizziness,easy,family_history,feel_worthwhile1,feel_worthwhile2,feel_worthwhile3,feel_worthwhile4,fun,indulgent,jointProblem,moderate_act,muscles,phys_activity,physicallyCapable,pleasurable,relaxing,satisfiedwith_life,sleep_diagnosis1,sleep_time,sleep_time1,social,unhealthy,vigorous_act,weight,work,Doctoral_Degree_PhD_MD_JD_etc,Grade_school,High_school_diploma,`Master's_Degree`,No_not_SpanishHispanicLatino,Some_college_or_vocational_school_or_Associate_Degree,Some_other_race,White,`White_and_Black_African-American_or_Negro`,Yes_other_Spanish_Hispanic_Latina,sugar_drinks,alcohol,fish,fruit,grains,vegetable,sodium]
            dfq = pdg.read_gbq("""select patient,test_date_mhc,atwork,alcohol,beneficial,body_remarkable_self_healing,body_self_healing_from_most_conditions_and_diseases,body_self_healing_in_many_different_circumstances,chestPain,chestPainInLastMonth,chronic_illness_body_betrayal,chronic_illness_body_blame,chronic_illness_body_coping,chronic_illness_body_failure,chronic_illness_body_handling,chronic_illness_body_management,chronic_illness_body_meaning,chronic_illness_challenge,chronic_illness_empowering,chronic_illness_handling,chronic_illness_impact,chronic_illness_management,chronic_illness_more_meaning_in_life,chronic_illness_positive_opportunity,chronic_illness_relatively_normal_life,chronic_illness_runing_life,chronic_illness_spoil,convenient,currentSmokeless,currentSmoking,currentVaping,disease,dizziness,easy,everQuitSmokeless,everQuitSmoking,everQuitVaping,family_history,feel_worthwhile1,feel_worthwhile2,feel_worthwhile3,feel_worthwhile4,fish,heartCondition,fruit,fun,grains,indulgent,jointProblem,moderate_act,muscles,onsetSmoking,onsetVaping,pastSmokeless,pastVaping,phys_activity,physicallyCapable,pleasurable,readinessQuitSmokeless,readinessQuitSmoking,readinessQuitVaping,relaxing,riskfactors1,riskfactors2,riskfactors3,riskfactors4,satisfiedwith_life,sleep_diagnosis1,sleep_time,sleep_time1,social,sugar_drinks,unhealthy,vegetable,vigorous_act,weight,work,CA,Cognitive_therapy__counselling,Cold_turkey,College_graduate_or_Baccalaureate_Degree,DE,Doctoral_Degree_PhD_MD_JD_etc,GB,Grade_school,High_school_diploma,`Master's_Degree`,Nicotine_replacement_product_patch_gum_lozenge_inhaler_nasal_spray,No_not_SpanishHispanicLatino,SE,SG,Some_college_or_vocational_school_or_Associate_Degree,US,Yes_Cuban,Yes_Mexican_Mexican_American_or_Chicano,Yes_other_Spanish_Hispanic_Latina from `imperial-410612.MHC_PH.questionnaire_clean_us2`
                               join (select patient,min(createdOn) test_date_mhc from `imperial-410612.MHC_PH.questionnaire_us2`
                                    group by patient) T using(patient)""",project_id=project_id)
            dfq.loc[:,'atwork':] = dfq.loc[:,'atwork':].astype(float).replace(-9999,np.nan) 
            
            # where variable = 'DistanceWalkingRunning'
            # df_6mwt_cl['test_date'] = pd.to_datetime(df_6mwt_cl['test_date'] + '0',format='%Y-w%U%w').dt.date
            df_6mwt_cl['test_date'] = pd.to_datetime(df_6mwt_cl['test_date'],utc=True).dt.date
            # df_6mwt_mhc['test_date'] = pd.to_datetime(df_6mwt_mhc['test_date'],utc=True).dt.date
            df_act_mhc['test_date'] = pd.to_datetime(df_act_mhc['test_date_mhc'],utc=True).dt.date
            # svae
            df_6mwt_cl.to_csv(f"data/6mwt_cl{suffix}.csv",index=False)
            # df_6mwt_mhc.to_csv("data/6mwt_mhc.csv",index=False)
            df_act_mhc.to_csv(f"data/act_mhc{suffix}.csv",index=False)
            dfq.to_csv(f"data/dfq{suffix}.csv",index=False)
        else:
            df_6mwt_cl = pd.read_csv(f"data/6mwt_cl{suffix}.csv")
            # df_6mwt_mhc = pd.read_csv("data/6mwt_mhc.csv")
            df_act_mhc = pd.read_csv(f"data/act_mhc{suffix}.csv")
            
            # ids = pdg.read_gbq('SELECT patient,diagnosis_date FROM `imperial-410612.MHC_PH.id_mapping2`',project_id=project_id)


        # df_fc['test_date'] = pd.to_datetime(df_act_mhc['test_date'],utc=True).dt.date

        # # tranform into months
        # df_6mwt_cl['test_month'] = pd.to_datetime(df_6mwt_cl['test_date']).dt.strftime('%Y-%B')
        # df_6mwt_mhc['test_month'] = pd.to_datetime(df_6mwt_mhc['test_date']).dt.strftime('%Y-%B')
        # df_act_mhc['test_month'] = pd.to_datetime(df_act_mhc['test_date']).dt.strftime('%Y-%B')
        # # df_fc['test_month'] = pd.to_datetime(df_act_mhc['test_date']).dt.strftime('%Y-%B')

        # # get the monthly average
        # clvaragg = {'who_fc':'mean', 'bnp_value':'mean', 'pbnp_value':'mean', '6mwt_value':'mean', 'iswt_value':'mean'}
        # df_6mwt_cl2 = df_6mwt_cl.groupby(['patient','Group','hospital','test_month']).agg(clvaragg).reset_index()
        # df_6mwt_mhc2 = df_6mwt_mhc.groupby(['patient','Group','test_month'])['distance'].mean().reset_index()
        # df_act_mhc2 = df_act_mhc.groupby(['patient','Group','test_month','variable'])['distance'].mean().reset_index()
        # # df_fc2 = df_fc.groupby(['patient','Group','test_month'])['who_fc'].mean().reset_index()

        # # merge by month
        # # df_6mwt_week = df_6mwt_cl.merge(df_6mwt_mhc,on=['patient','Group','test_week']).merge(df_act_mhc,on=['patient','Group','test_week'])
        # df_6mwt_month = df_6mwt_cl2.merge(df_6mwt_mhc2,on=['patient','Group','test_month'],how='left').merge(df_act_mhc2,on=['patient','Group','test_month'],how='left')
        # # df_6mwt_month = df_6mwt_month.merge(df_fc2,on=['patient','Group','test_month'],how='left')
        # cols = ['patient', 'Group', 'hospital', 'distance_x','distance_y','test_month','variable'] + list(clvaragg.keys())
        # df_6mwt_month = df_6mwt_month.loc[:,cols]
        # # df_6mwt_month.loc[:,'distance_x'] = df_6mwt_month.loc[:,'distance_x'].clip(0,650)
        # df_6mwt_month.rename(columns={"distance_x":"6mwt_mhc","distance_y":"act"},inplace=True)
        # df_6mwt_month['aggregation'] = 'month'
        # # pdg.to_gbq(df_6mwt_month.astype(str),"MHC_PH.6mwt_all_month", project_id=project_id,if_exists="replace")

        # # get the patient average
        # df_6mwt_cl2 = df_6mwt_cl.groupby(['patient','Group','hospital']).agg(clvaragg).reset_index()
        # df_6mwt_mhc2 = df_6mwt_mhc.groupby(['patient','Group'])['distance'].mean().reset_index()
        # df_act_mhc2 = df_act_mhc.groupby(['patient','Group','variable'])['distance'].mean().reset_index()
        # # df_fc2 = df_fc.groupby(['patient','Group'])['who_fc'].mean().reset_index()
        # ## merge by patient only
        # df_6mwt_patient = df_6mwt_cl2.merge(df_6mwt_mhc2,on=['patient','Group'],how='left').merge(df_act_mhc2,on=['patient','Group'],how='left')
        # # df_6mwt_patient = df_6mwt_patient.merge(df_fc2,on=['patient','Group'],how='left')
        # # df_6mwt_patient.loc[:,'distance_x'] = df_6mwt_patient.loc[:,'distance_x'].clip(0,650)
        # df_6mwt_patient.rename(columns={"distance_x":"6mwt_mhc","distance_y":"act"},inplace=True)
        # df_6mwt_patient['aggregation'] = 'patient'
        # df_6mwt_patient = pd.concat([df_6mwt_month,df_6mwt_patient],axis=0,ignore_index=True)
        # pdg.to_gbq(df_6mwt_patient.astype(str),"MHC_PH.6mwt_all_patient", project_id=project_id,if_exists="replace")
        # df = pdg.read_gbq("MHC_PH.6mwt_all_patient", project_id=project_id)
        # # now merge the data for functional class and

        # ##### this "SPVDU01314" patient has a huge variability
        # # transform 6mwt_mhc as variable
        # idx = df_6mwt_patient['6mwt_mhc'].notna()
        # aux = df_6mwt_patient.loc[idx,['patient', 'Group',  'hospital','test_month','6mwt_mhc','who_fc',
        #                                                                 'bnp_value', 'pbnp_value', '6mwt_value', 'iswt_value','aggregation']].drop_duplicates()
        # aux['variable'] = '6mwt_mhc'
        # aux = aux.rename(columns={'6mwt_mhc':'act'})
        # df_6mwt_patient.drop(index=np.where(idx)[0],inplace=True)
        # df_6mwt_patient = pd.concat([df_6mwt_patient,aux],axis=0,ignore_index=True)
        # df_6mwt_patient.drop(columns='6mwt_mhc',inplace=True)
        # pdg.to_gbq(df_6mwt_patient.astype(str),"MHC_PH.6mwt_all_patient_f", project_id=project_id,if_exists="replace")

        def get_day(x):
            try:
                day = x.days
            except:
                print('failed',x)   
                day = np.nan
            return day
        
        def add_row(df,row):
            for v,r in row.items():
                if v not in ('patient'):
                    df[v] = r
            return df

        # add mhc 6mwt as variable
        # df_6mwt_mhc['variable'] = "6mwt MHC"
        # df_act_mhc2 =  pd.concat([df_act_mhc,df_6mwt_mhc],axis=0,ignore_index=True).rename(columns={'distance':'act','test_date':'test_date_mhc'}).dropna(subset=['test_date_mhc'])
        # aggregate by month I have done by 6 months!! and median
        if agg_lvl == 'quarter':
            df_act_mhc['test_date_mhc'] =pd.PeriodIndex(pd.to_datetime(df_act_mhc['test_date_mhc']).dt.to_period('Q'), freq='Q').to_timestamp()
        elif agg_lvl == 'month':
            df_act_mhc['test_date_mhc'] = pd.to_datetime(pd.to_datetime(df_act_mhc['test_date_mhc']).dt.strftime('%Y-%B')).dt.date

        df_act_mhc = df_act_mhc.groupby(['patient','test_date_mhc','device','variable'])['act'].agg(aggregation).reset_index()

        # filter by post diagnosis only
        # df_6mwt_cl = df_6mwt_cl.merge(ids,on='patient',how='left')
        # df_6mwt_cl = df_6mwt_cl.query('test_date < diagnosis_date').reset_index(drop=True)
        # df_6mwt_cl['test_date'] = pd.to_datetime(df_6mwt_cl['test_date'])
        # df_act_mhc2['test_date_mhc'] = pd.to_datetime(df_act_mhc2['test_date_mhc'])

        # aggregate by patient
        # df_act_mhc2 = df_act_mhc2.groupby(['patient','variable']).agg({'act':'mean','test_date_mhc':'max'}).reset_index()

        # remove duplicates
        df_6mwt_cl = df_6mwt_cl.drop_duplicates().dropna(subset=['who_fc',  'pbnp_score',  'walk_score',  'pbnp_value',  '6mwt_value',  'iswt_value',  'risk_score'],how='all',inplace=False).reset_index(drop=True)

        # prepare data for questionnaire - activity correlations
        df_act_mhc0 = df_act_mhc.copy()
        df_act_mhc0['variable_device'] = df_act_mhc0['device'] + df_act_mhc0['variable']
        df_act_mhc0 = df_act_mhc0.pivot_table(columns='variable',values='act',index=['patient','test_date_mhc']).reset_index()
        corrs_act_q = df_act_mhc0.drop(columns=['test_date_mhc']).merge(dfq.drop(columns=['test_date_mhc']),on='patient')
        corrs_act_q = corrs_act_q.iloc[:,1:].corr()
        corrs_act_q = corrs_act_q.iloc[:df_act_mhc0.shape[1]-2,dfq.shape[1]-2:]

    # find the closest date
        A = []
        for ri, row in df_6mwt_cl.iterrows():
            
            if row.isna()['test_date']:
                continue

            idx = np.where(df_act_mhc['patient'] == row['patient'])[0]
            aux = df_act_mhc.loc[idx,['patient','test_date_mhc','device','variable','act']]
            aux = add_row(aux,row)
            
            aux['date_diff'] = (pd.to_datetime(aux['test_date_mhc']) - pd.to_datetime(aux['test_date'])).apply(get_day).abs()

            # order and keep the first only
            aux = aux.sort_values(by=['patient','variable','device','date_diff'],ascending=[True,True,True,True])
            aux.drop_duplicates(subset=['patient','variable','device','test_date'],inplace=True,keep='first')
            # aux.drop(columns=['date_diff'],inplace=True)
            A.append(aux)    
            # df_6mwt_all = aux.copy() if ri == 0 else df_6mwt_all.merge(aux,on=['patient','test_date','variable'],how='outer')
        df_6mwt_all = pd.concat(A,axis=0,ignore_index=True)

        # add questionnaire data
        dfq = pd.melt(dfq,id_vars=['patient','test_date_mhc'],var_name='variable',value_name='act')
        dfq = df_6mwt_cl.merge(dfq,on='patient')
        df_6mwt_all = pd.concat([df_6mwt_all,dfq],axis=0)
        
        df_6mwt_all.loc[:,'who_fc':'risk_score'] = df_6mwt_all.loc[:,'who_fc':'risk_score'].astype(float)
        df_6mwt_all = df_6mwt_all.drop_duplicates(subset=['patient','variable','act','who_fc',
                            'pbnp_score',  'walk_score',  
                            'pbnp_value',  '6mwt_value',  'iswt_value',  'risk_score']).reset_index(drop=True)

        # calculate correlations - activity  + questionnaire with clinical
        corr = {}
        D = []
        
        for var in df_6mwt_all['variable'].unique():
            if var == "High_school_diploma":
                a = 0.
            if str(var) == 'nan':
                continue
            for var2 in df_6mwt_all.loc[:,'who_fc':'risk_score'].columns:
                dfnow = df_6mwt_all.query(f'variable == "{var}"').dropna(subset=['act',var2],how='any').loc[:,['patient','act',var2,'date_diff']]#.drop_duplicates(subset=['patient','act',var2])
                dfnow.query('date_diff <= 182 or date_diff!=date_diff',inplace=True)
                if dfnow.shape[0] < 10:
                    corr.update({var+'-'+var2:[np.nan,np.nan,dfnow.shape[0],dfnow.loc[:,'patient'].nunique()]})
                    print(var+'-'+var2,corr[var+'-'+var2])
                else:
                    presult = pearsonr(dfnow.loc[:,'act'].values,dfnow.loc[:,var2].values)
                    corr.update({var+'-'+var2:[presult.statistic,presult.pvalue,dfnow.shape[0],dfnow.loc[:,'patient'].nunique()]})
                    print(var+'-'+var2,corr[var+'-'+var2])

            corr_df = pd.DataFrame(corr, index=['correlation','pvalue','n','n_patients']).T.reset_index()
            corr_df.columns = ['variable', 'correlation','pvalue','n','n_patients']
            corr_df['correlation_abs'] = np.abs(corr_df['correlation'])
            corr_df = corr_df.sort_values(by='correlation_abs',ascending=False)
            corr_df.dropna(subset=['correlation'],inplace=True)
            # print('correlations:',corr_df.loc[:,['variable','correlation']])

            # dfnow = df_6mwt_patient.query(f'`6mwt_mhc` == `6mwt_mhc` and aggregation == "{agg}"').drop_duplicates(subset=['6mwt_cl','6mwt_mhc'])
            # cnow = dfnow.loc[:,['6mwt_mhc','6mwt_cl']].corr().iloc[0,1]
            # print(f'correlations 6mwt {agg}:',cnow)
            D.append(corr_df)
        D = pd.concat(D,axis=0,ignore_index=True)
        D.to_csv(f'data/correlations_new2_{agg_lvl}_{aggregation}.csv',index=False)

        corrs_act_q = corrs_act_q.reset_index().melt(id_vars=['index'])
        corrs_act_q['variable'] = corrs_act_q['index'] +'-'+ corrs_act_q['variable']
        corrs_act_q.drop(columns=['index'],inplace=True)
        corrs_act_q.to_csv(f'data/correlations_q_act{suffix}.csv',index=False)
        print('walktest data recomputed')

        # return D,corrs_act_q
    else:
        print('walktest data ignored!') 



# # tranform into quarters - it is not working well
# df_6mwt_cl['test_quarter'] = pd.to_datetime(df_6mwt_cl['test_date']).dt.strftime('%Y-Q')+pd.to_datetime(df_6mwt_cl['test_date']).dt.quarter.astype('string')
# df_6mwt_mhc['test_quarter'] = pd.to_datetime(df_6mwt_mhc['test_date']).dt.strftime('%Y-Q')+pd.to_datetime(df_6mwt_mhc['test_date']).dt.quarter.astype('string')
# df_act_mhc['test_quarter'] = pd.to_datetime(df_act_mhc['test_date']).dt.strftime('%Y-Q')+pd.to_datetime(df_act_mhc['test_date']).dt.quarter.astype('string')

# # get the quarterly average
# df_6mwt_cl2 = df_6mwt_cl.groupby(['patient','Group','hospital','test_quarter'])['distance'].mean().reset_index()
# df_6mwt_mhc2 = df_6mwt_mhc.groupby(['patient','Group','test_quarter'])['distance'].mean().reset_index()
# df_act_mhc2 = df_act_mhc.groupby(['patient','Group','test_quarter'])['distance'].mean().reset_index()
# # merge by quarterly
# df_6mwt_quarter = df_6mwt_cl2.merge(df_6mwt_mhc2,on=['patient','Group','test_quarter'],how='left').merge(df_act_mhc2,on=['patient','Group','test_quarter'],how='left')
# cols = ['patient', 'Group', 'hospital', 'distance_x','distance_y','distance','test_quarter']
# df_6mwt_quarter = df_6mwt_quarter.loc[:,cols]
# df_6mwt_quarter.loc[:,'distance_x'] = df_6mwt_quarter.loc[:,'distance_x'].clip(0,650)
# df_6mwt_quarter.rename(columns={"distance_x":"6mwt_cl","distance_y":"6mwt_mhc","distance":"act"},inplace=True)
# pdg.to_gbq(df_6mwt_quarter.astype(str),"MHC_PH.6mwt_all_quarter", project_id=project_id,if_exists="replace")

def prepare_data_collated_dist(df2,slopes):
    ###### all collated distributions - the plot contains all the variables, pvalue and the slopes prediagnosis
    # scale
    df22 = df2.copy()

    scaler = StandardScaler()
    minmaxscaler = MinMaxScaler()
    for v,d in df22.loc[:,['variable','device']].drop_duplicates().to_numpy():
        aux = df22.query(f"variable == '{v}' and device == '{d}'")
        
        # eliminate top 10% 
        thr = np.quantile(aux['value'].dropna(),.995)
        aux = df22.query(f"value < {thr}")

        aux.loc[:,'value_scaled'] = scaler.fit_transform(aux['value'].values.reshape(-1,1))
        aux.loc[:,'value_minmax'] = minmaxscaler.fit_transform(aux['value'].values.reshape(-1,1))
        
        df22.loc[aux.index,'value_scaled'] = aux['value_scaled']
        df22.loc[aux.index,'value_minmax'] = aux['value_minmax']

    df22['value_logged'] = df22['value'].apply(lambda x: np.log(x) if x > 0 else 0)

    df22['unit_labels'] = df22['variable'].map(unit_labels)
    df22['variable'] = df22['variable'].map(rename_dict)

    # get the prediagnosis data only - after computing df2now
    # df22 = df22.query('Group != "PH" or variable in ("Asleep","Awake","InBed>18hr") or months_diagnosis <= 0').reset_index(drop=True)

    idx = df22.query("variable in ('StepCount','Basal Energy')").index
    df22.loc[idx,'unit_labels'] += ' x1000'
    units = df22.drop_duplicates(subset=['variable','unit_labels']).loc[:,['variable','unit_labels']]
    units['unit_labels'].fillna('hrs',inplace=True)

    if len(slopes)>0:
        slopes['variable'] = slopes['variable'].map(rename_dict)
        slopes = slopes.dropna(subset=['variable']).reset_index(drop=True)
        slopes['yu'] = 0.02
        slopes['yl'] = -0.02
        slopes['y'] = 0

    # slopes['time_segment_num'] = -slopes['time_segment'].apply(lambda x:x.split('-')[1]).astype(int)
    df2now = df2.copy().drop(columns=['months']).rename(columns={"months_diagnosis":"months"},inplace=False)
    df2now = df2now.groupby(['variable','Group','device','months'])['value'].agg(['mean','std','count']).reset_index()
    df2now['ci'] = 1.96*df2now['std']/np.sqrt(df2now['count'])
    df2now.columns = ['variable','Group','device','months','mean','std','count','ci']
    df2now['y'] = 0
    df2now['upper'] = df2now['mean'] + df2now['ci']
    df2now['lower'] = df2now['mean'] - df2now['ci']
    df2now['variable'] = df2now['variable'].map(rename_dict)
    df2sleep = df2now.copy().query('variable in ("Awake","Asleep","InBed")')

    idx = df2now.query("variable in ('StepCount','Basal Energy')").index
    df2now.loc[idx,['mean','upper','lower']] /= 1000

    # remove iPhone variables not of interest
    df2now = df2now.query('device != "iPhone" or (device == "iPhone" and variable not in ("Basal Energy","InBed>18hr"))').reset_index(drop=True)
    df22 = df22.query('device != "iPhone" or (device == "iPhone" and variable not in ("Basal Energy","InBed>18hr"))').reset_index(drop=True)
    df2now = df2now.query('months<12').reset_index(drop=True)
    df22 = df22.query('months<12').reset_index(drop=True)

    return df22,df2now,df2sleep,slopes,units


def statistical_analysis(df2,slopes,stat_analysis=True,freq_analysis=False,plot_flag=True):
    if stat_analysis:
        df2.dropna(subset=["variable"],inplace=True)
        colstokeep = ['Awake','Asleep','BedBound','VO2Max', 'HeartRate', 'HeartRateReserve',
                    'StepCount', 'AppleStandTime', 'FlightsClimbed','StepCountPaceMax','FlightsClimbedPaceMax',
                    'StepCountPaceMean','FlightsClimbedPaceMean',
                    'StepCountPaceMean','FlightsClimbedPaceMean','CardiacEffort',
                    'RestingHeartRate', 'BasalEnergyBurned', 'ActiveEnergyBurned', 'WalkingHeartRateAverage',
                    'HeartRateVariabilitySDNN']
        # merge Healthy and DC
        df2 = df2.query('variable in @colstokeep')
        df2 = df2.dropna(subset='value').reset_index(drop=True) 

        dfnow = df2.copy()

        dfnow['prediagnosis'] = (dfnow['months']<1).map({True:'prediagnosis',False:'postdiagnosis'})

        L = list(dfnow.groupby(['variable','device']))
        effects = pd.DataFrame([])

        for li,l in L:
            print(li)
            try:
                model = smf.mixedlm(
                    'value ~ ' + " + ".join(['age','bmi','gender','ethnicity','months','prediagnosis']),
                    #                 f'value ~ {covariate}',#+ bmi + gender + ethnicity
                    data = l.reset_index(drop=True).dropna(subset=['age','bmi','gender','ethnicity']),
                    groups='Group',
                    # re_formula='patient'
                )

                # Fit the model
                results = model.fit()
                print(li,results.summary())
                aux = pd.DataFrame(results.random_effects).T.rename(columns={'Group':'coeff'})
                aux['type'] = 'random_effect'
                a1 = pd.DataFrame(results.params).rename(columns={0:'coeff'})
                a1['type'] = 'fixed_effect'

                ##### scale correctly - get pvalues and stdE
                intercept = a1.loc['Intercept','coeff']
                a1.loc[:,'coeff'] /= intercept
                aux.loc[:,'coeff'] /= intercept 
                aux = pd.concat([a1,aux],axis=0)

                a2 = pd.DataFrame(results.pvalues).rename(columns={0:'pvals'})
                aux = pd.concat([aux,a2],axis=1)

                aux['variable'] = li[0]
                aux['device'] = li[1]
                aux['covariate'] = 'all'
                effects = pd.concat([effects,aux],axis=0)
            except Exception as e:
                print(e)

        effects = effects.reset_index()
        effects2 = effects.query('index == "Group Var" and abs(coeff)>0.01')
        effects2['coeff_abs'] = effects2['coeff'].abs()
        effects2 = effects2.sort_values('coeff_abs',ascending=False)

        effects3 = effects.query('index == "PH"').loc[:,['coeff','variable','device']].rename(columns={'coeff':'PH_RandEff'})

        effects2 = effects2.merge(effects3,on=['variable','device'])
        col_ord = ['variable', 'device','coeff', 'PH_RandEff', 'pvals']
        effects2 = effects2.loc[:,col_ord]
        effects.to_csv('data/effects4.csv',index=True)
        effects2.to_csv('data/effects4_simple.csv',index=True)
    else: 
        effects = pd.read_csv('data/effects4.csv')



        ###### need to estimate pvalue of random effect - re/intercept ± stdE/intercept
    if freq_analysis:

        pvals_df = pd.DataFrame()
        
        for ri,row in dfnow.groupby(['variable','device'])['value'].count().reset_index().query('value > 200').iterrows(): 
            var,device = row['variable'],row['device']
            print(var)
            dfnow2 = dfnow.query(f"variable == '{var}' and device == '{device}'")
            L = list(dfnow2.groupby("Group"))
            lv = np.array([ll[0] for ll in L])
            def find_idx(lv,v):
                try:
                    return np.where(lv == v)[0][0]
                except:
                    return None
                
            lv_idx = [find_idx(lv,v) for v in ['PH','DC','Healthy']]
            group1 = L[lv_idx[0]][1]["value"]
            group3 = L[lv_idx[2]][1]["value"]
            pvals_df = calculate_all_pvals(pvals_df,var,device,group1, group3,L,lv_idx,name='PH-Healthy',idxs=[0,2])
            try:
                group2 = L[lv_idx[1]][1]["value"]
                pvals_df = calculate_all_pvals(pvals_df,var,device,group1, group2,L,lv_idx,name='PH-DC',idxs=[0,1]) 
            except Exception as e:
                print(e)

            # pvals_df = calculate_all_pvals(pvals_df,var,device,group2, group3,L,lv_idx,name='DC-Healthy',idxs=[1,2]) 
            # res = ks_2samp(L[0][1], 
            #      L[1][1], 
            #      alternative='less', 
            #      method='auto')
            # wilcoxon was not used because they have unequal lengths - and the assumption of normality is not met
            # this is the unadjusted p-value
        pvals_df.to_csv('data/pvals_final_binary.csv',index=False)
        # pvals_df.to_csv('data/pvals_final.csv',index=False)# this is for PH-Rest
        
    else:
        pvals_df = pd.read_csv('data/pvals_final_binary.csv')

    if plot_flag:
        # prepare data
        df22,df2now,df2sleep,slopes,units = prepare_data_collated_dist(df2,slopes)
        
        # plot it
        allvar_plot(slopes.copy(),df22.copy(),df2now.copy(),units)

        sleep_plot(df22.copy(),df2sleep,units)
        
    else: 
        print('Frequentist analysis ignored!')
        
def extract_features(df2,cohort='UK',extract_feat_flag=False):
    # daily aggregations have been removed!!! maybe it is worth re-introducing them
    logic = "" if cohort == "UK" else " not" 
    dfq = pdg.read_gbq(f"""select *
                       FROM `imperial-410612.MHC_PH.questionnaire_clean_us2`
                            where patient{logic} like "SPVDU%" """,project_id=project_id)
    # dfq = pdg.read_gbq(f"""select patient,atwork,alcohol,beneficial,body_remarkable_self_healing,body_self_healing_from_most_conditions_and_diseases,body_self_healing_in_many_different_circumstances,chestPain,chestPainInLastMonth,chronic_illness_body_betrayal,chronic_illness_body_blame,chronic_illness_body_coping,chronic_illness_body_failure,chronic_illness_body_handling,chronic_illness_body_management,chronic_illness_body_meaning,chronic_illness_challenge,chronic_illness_empowering,chronic_illness_handling,chronic_illness_impact,chronic_illness_management,chronic_illness_more_meaning_in_life,chronic_illness_positive_opportunity,chronic_illness_relatively_normal_life,chronic_illness_runing_life,chronic_illness_spoil,convenient,currentSmokeless,currentSmoking,currentVaping,disease,dizziness,easy,everQuitSmokeless,everQuitSmoking,everQuitVaping,family_history,feel_worthwhile1,feel_worthwhile2,feel_worthwhile3,feel_worthwhile4,fish,fruit,fun,grains,indulgent,jointProblem,heartCondition,moderate_act,muscles,onsetSmoking,onsetVaping,pastSmokeless,pastVaping,phys_activity,physicallyCapable,pleasurable,readinessQuitSmokeless,readinessQuitSmoking,readinessQuitVaping,relaxing,riskfactors1,riskfactors2,riskfactors3,riskfactors4,satisfiedwith_life,sleep_diagnosis1,sleep_time,sleep_time1,social,sugar_drinks,unhealthy,vegetable,vigorous_act,weight,work,CA,Cognitive_therapy__counselling,Cold_turkey,College_graduate_or_Baccalaureate_Degree,DE,Doctoral_Degree_PhD_MD_JD_etc,GB,Grade_school,High_school_diploma,`Master's_Degree`,Nicotine_replacement_product_patch_gum_lozenge_inhaler_nasal_spray,No_not_SpanishHispanicLatino,SE,SG,Some_college_or_vocational_school_or_Associate_Degree,US,Yes_Cuban,Yes_Mexican_Mexican_American_or_Chicano,Yes_other_Spanish_Hispanic_Latina 
    #                    FROM `imperial-410612.MHC_PH.questionnaire_clean_us2`
    #                         where patient{logic} like "SPVDU%" """,project_id=project_id)
    dfq.iloc[:,1:] = dfq.iloc[:,1:].astype(float).replace(-1,np.nan) 
    # df2['variable-device'] = df2['variable']+df2['device']
    df2['cohort'] = cohort

    if extract_feat_flag:
        
        df_features_all = []

        for mode in ['all','prediagnosis','postdiagnosis']:#,'6months_prediagnosis','1year_prediagnosis'
            if mode == 'all':
                df = df2.copy() # averaged by month
                # df_ = df2.copy() # averaged by month
            elif mode == 'prediagnosis':
                df = df2.query('months_diagnosis<=0 or cohort=="US" or Group != "PH"').copy() # averaged by day
                # df_ = df2.query('months_diagnosis<=0 or cohort=="US" or Group != "PH"').copy() # averaged by month
            elif mode == 'postdiagnosis':
                df = df2.query('months_diagnosis>0 or cohort=="US" or Group != "PH"').copy() # averaged by day
                # df_ = df2.query('months_diagnosis>0 or cohort=="US" or Group != "PH"').copy() # averaged by month
            elif mode == '6months_prediagnosis':
                df = df2.query('months_diagnosis<0 and months_diagnosis>-6').copy() # averaged by day
                # df_ = df2.query('months_diagnosis<0 and months_diagnosis>-6').copy() # averaged by month
            elif mode == '1year_prediagnosis':
                df = df2.query('months_diagnosis<-12').copy() # averaged by day
                # df_ = df2.query('months_diagnosis<-12').copy() # averaged by month
            
            # extract timeseries features
            df_features = pd.DataFrame()
            for i,v in df.groupby(["patient","variable","device"]):
                aux = stat_feature_extraction(v,colname="value")
                aux1 = fft_feature_extraction(v,colname="value")
                # aux1 = stat_feature_extraction(df_.query(f"patient == '{i[0]}' and `variable-device` == '{i[1]}'"),colname="value")
                # aux1['features'] = aux1['features'].apply(lambda x:x+"_mon")
                aux['feat_ext_type'] = 'stat'
                aux1['feat_ext_type'] = 'fft'
                # ARIMA features
                try:
                    aux2  = ARIMA_feature_extraction(i,v)
                except Exception as e:
                    print(e)
                    aux2 = pd.DataFrame([])
                aux2['feat_ext_type'] = 'ARIMA'
                aux = pd.concat([aux,aux1,aux2],axis=0,ignore_index=True)

                aux["patient"] = i[0]
                aux["variable"] = i[1]
                aux["device"] = i[2]
                
                df_features = pd.concat([df_features,aux],
                                        axis=0)

            # cross correlation between each combination of variables
            dff = df.copy()
            dff['variable'] = dff['variable']+dff['device']
            aux3 = CC_feature_extraction(dff)
            # aux3['device'] = aux3['device'].fillna('both')
            aux3['feat_ext_type'] = 'CC'
            df_features = pd.concat([df_features,aux3],axis=0)
            df_features['mode'] = mode
            df_features_all.append(df_features)
        df_features_all = pd.concat(df_features_all,axis=0,ignore_index=True)
        pdg.to_gbq(df_features_all,f"MHC_PH.features{cohort}", project_id="imperial-410612",if_exists="replace")
    else:
        # df_features_all = pdg.read_gbq(f"select * from `imperial-410612.MHC_PH.featuresUK-2024-12-04T15_05_43`", project_id="imperial-410612")
        df_features_all = pdg.read_gbq(f"select * from MHC_PH.features{cohort}", project_id="imperial-410612")
        
    
    dfq = dfq.fillna(-1)
    
    return dfq,df_features_all

def ml_pipeline(trainset,testset,ml_pipe_flag=False,external_flag = '',lbl = 'all',finetune_test_size=.3,feature_spaces = ['activity','questionnaire','activity_questionnaire']):
    if not ml_pipe_flag:
        print(f'ML pipeline {external_flag} did not run')
        return ''
    
    if external_flag not in ['','_external','_finetune']:
        print("wrong external_flag method, implemented methods are: '','_external' or '_finetune'")
        print(f'ML pipeline {external_flag} did not run')
        return ''
    
    name_output_file = "8_"+str(np.round(finetune_test_size,1)) if external_flag=='_finetune' else "8"
    name_output_file += lbl  

    order_model = 0 # select the middle model
    # suffix = '_external' if external_flag else ''

    dfq,df_features_all = trainset
    dfq_us,df_features_all_us = testset

    ####### patient has been removed!!!!!
    # df2 = pd.concat([df2,df2_us.query('patient != "04ee62df-475b-477b-ad74-5a206d8a1946"')],axis=0)
    df_id_mapping = pdg.read_gbq('select patient,`Group` from `imperial-410612.MHC_PH.id_mapping2` where patient != "04ee62df-475b-477b-ad74-5a206d8a1946"',project_id=project_id).drop_duplicates()
    df_id_mapping['Group'] = df_id_mapping['Group'].map({'Not found':'Not found', 'Healthy':'Healthy', 'PAH':'PH', 'COVID':'DC', 'NotPH':'DC', 'PH5':'PH','CV':'DC','CTEPH':'PH'})

    dfq = pd.concat([dfq,dfq_us],axis=0)
    
    params = pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/bayesian_hyperparameters.csv')
    RESULTS = []
    roc_df = pd.DataFrame([])
    for mode in ['prediagnosis','postdiagnosis']:
    # for mode in ['prediagnosis']:
        # [114247 rows x 7 columns]
        # [446651 rows x 5 columns]
        df_features = pd.concat([reformat_features(df_features_all,mode),reformat_features(df_features_all_us,mode)],axis=0)

        # for gi,groups in enumerate([('PH','Healthy','DC'),('PH','Healthy'),('PH','DC')]):
        for gi,groups in enumerate([('PH','Healthy','DC')]):
            for feature_space in feature_spaces:
                
            
                print('Analysing...',mode,feature_space,groups)
                # if gi == 0 and feature_space == 'activity_questionnaire' and mode == 'prediagnosis':
                #     a = 7
                # try:
                #     parsnow = params.query(f'mode == "{mode}" and feature_space == "{feature_space}" and group == "{groups[0]}_{groups[1]}"').loc[:,['parameter',	'value']].set_index('parameter').to_dict()['value']
                #     # make all of them integers except learning rate
                #     parsnow['max_depth'] = int(parsnow['max_depth'])
                #     parsnow['n_estimators'] = int(parsnow['n_estimators'])
                #     parsnow['reg_alpha'] = int(parsnow['reg_alpha'])
                #     parsnow['scale_pos_weight'] = int(parsnow['scale_pos_weight'])
                # except:
                # parsnow = {'reg_alpha':2,'reg_lambda':1,'scale_pos_weight':2**(-1)**(1-gi),'learning_rate':.07, 'max_depth':6, 'n_estimators':30}
                parsnow = {'reg_alpha':2,'reg_lambda':5,'scale_pos_weight':2**(-1)**(1-gi),'learning_rate':.07, 'max_depth':6, 'n_estimators':30}
                                            
                ##### feature space selection
                if feature_space == 'activity_questionnaire':
                    df_featuresnow = df_features.reset_index().merge(dfq,on='patient',how='outer')
                    df_featuresnow.index = df_featuresnow['patient']
                    df_featuresnow = df_featuresnow.drop(columns=['patient'])

                elif feature_space == 'activity':
                    df_featuresnow = df_features.copy()
                    
                elif feature_space == 'questionnaire':
                    df_featuresnow = dfq.copy()
                    df_featuresnow.index = df_featuresnow['patient']
                    df_featuresnow = df_featuresnow.drop(columns=['patient'])

                # extract x and y
                X = df_featuresnow.replace('nan',np.nan).to_numpy().astype(float)
                # X = df_featuresnow.fillna(-9999).to_numpy()
                # aux = df2.loc[:,["Group","patient"]].drop_duplicates().reset_index(drop=True)
                aux = df_id_mapping.merge(pd.Series(df_featuresnow.index),how="right",on="patient")#.dropna(subset='Group')
                y = (aux["Group"] == "PH").values

                # select only PH & Healthy
                idx = aux["Group"].isin(groups)
                y = y[idx]
                X = X[idx,:]

                ################ Supervised Learning ###################

                # Generate some example data
                np.random.seed(30)

                # Split the data into training and testing sets
                idx_uk = aux.loc[idx,"patient"].str.startswith('SPVDU')
                if external_flag=='_external':
                    X_train, X_test, y_train, y_test = X[idx_uk],X[~idx_uk],y[idx_uk],y[~idx_uk]
                elif external_flag=='_finetune':
                    X_test, X_train_leak, y_test, y_train_leak = train_test_split(X[~idx_uk], y[~idx_uk], test_size=finetune_test_size, random_state=88)
                    X_train = np.concatenate([X[idx_uk],X_train_leak],axis=0)
                    y_train = np.concatenate([y[idx_uk],y_train_leak],axis=0)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X[idx_uk], y[idx_uk], test_size=0.3, random_state=88)
                
                
                # scale features
                scaler = StandardScaler()
                # X_train = scaler.fit_transform(X_train)
                # X_test = scaler.fit_transform(X_test)
                # X = scaler.fit_transform(X)

                # Initialize the XGBoost classifier
                xgb_classifier = XGBClassifier(objective='binary:logistic', 
                                            random_state=30,
                                            **parsnow)

                model = make_pipeline(scaler, xgb_classifier)

                # Train the classifier on the training data
                # model.fit(X_train, y_train)

                # search hyperparameters by Bayesian search
                # xgb_classifier = bayesian_search(X,y)

                # Make predictions on the test set - with crossvalidation
                # y_pred = xgb_classifier.predict(X_test)
                # y_pred_cv = cross_val_predict(model, X_train, y_train, cv=5)
                xgboostscores = cross_validate(model, X_train, y_train, scoring=['f1','roc_auc'], cv=5, return_estimator=True)
                # order_model = -1 if mode == 'all' and "_".join(groups) == "PH_Healthy" else -2
                idx_better_than_random = xgboostscores['test_roc_auc']>=.5
                estimators = np.arange(5)[idx_better_than_random] if idx_better_than_random.sum()!=0 else np.arange(5)
                worst_xgb = xgboostscores['estimator'][xgboostscores['test_roc_auc'][estimators].argsort()[order_model]]#[xgboostscores['test_roc_auc'].argmax()]

                # results = evaluate_classification_complete(xgb_classifier,X_test,y_test,y_pred,name="xgboost",save_flag = True)
                threshold = .5 if mode == 'all' else .5
                # path = best_model[1].cost_complexity_pruning_path(X_train, y_train)
                # ccp_alphas, impurities = path.ccp_alphas, path.impurities
                results_cv_df = evaluate_classification_wrapper(gi,worst_xgb,X_train,y_train,feature_space,groups,mode,dataset='train',threshold = 'optimize',name="xgboost",save_flag = False)
                RESULTS.append(results_cv_df)
                results_cv_df = evaluate_classification_wrapper(gi,worst_xgb,X_test,y_test,feature_space,groups,mode,dataset='test',threshold = results_cv_df['threshold'].values[0],name="xgboost",save_flag = False)
                RESULTS.append(results_cv_df)
                xgb_roc_auc = results_cv_df['roc_auc']['50%']
                
                ###### plot tree
                # xgb_classifier.get_booster().feature_names = list(df_features.keys())
                # plot_tree(xgb_classifier,  num_trees=1)
                # plt.show()

                # Extract feature importance scores
                feature_importance = worst_xgb[1].feature_importances_

                # Create a DataFrame to display feature importance
                feature_importance_df = pd.DataFrame({'features': list(df_featuresnow.keys()),
                                                    'importances': feature_importance})

                # Sort the DataFrame by importance in descending order
                feature_importance_df = feature_importance_df.sort_values(by='importances', ascending=False).reset_index(drop=True)
                feature_importance_df["importances_cum"] = feature_importance_df["importances"].cumsum()

                # Display the feature importance
                print("Feature Importance:")
                print(feature_importance_df.iloc[:9,:])
                feature_importance_df["date"] = datetime.now().strftime("%Y-%m-%d")
                feature_importance_df["model"] = f"naive models FINAL{name_output_file}{external_flag}"
                feature_importance_df['feature_space'] = feature_space
                feature_importance_df['groups'] = "_".join(groups)
                feature_importance_df['mode'] = mode
                feature_importance_df['n_ratio'] = str(y.sum())+":"+str(len(y)-y.sum())
                pdg.to_gbq(feature_importance_df,"MHC_PH.feature_importance2",project_id=project_id,if_exists="append")

                # Fits the explainer
                # explainer = shap.Explainer(best_model[1].predict, X_train)

                # Calculates the SHAP values - It takes some time
                # shap_values = explainer(X_train)

                # plt.figure()
                # shap.summary_plot(shap_values)
                # plt.savefig(f'shap_{"".join(groups)}_{feature_space}_{mode}.png')
                # shap.plots.bar(shap_values[0])
                # shap.plots.waterfall(shap_values[0])

                # xgb_classifier.save_model(f'models/xgboost_May24_{feature_space}_{mode}_{"_".join(groups)}.json')

                ######### now repeat for SVM
                # Initialize the SVM classifier
                svm_classifier = SVC(kernel='linear', random_state=30,probability=True)
                model = make_pipeline(scaler, svm_classifier)
                # train the model
                # svm_classifier.fit(X_train, y_train)
                # Make predictions on the test set - with crossvalidation
                svm_scores = cross_validate(model, np.nan_to_num(X_train,nan=-1), y_train, scoring=['f1','roc_auc'], cv=5, return_estimator=True)
                # order_model = -1 if mode == 'all' and "_".join(groups) == "PH_Healthy" else -2
                idx_better_than_random = svm_scores['test_roc_auc']>=.5
                estimators = np.arange(5)[idx_better_than_random] if idx_better_than_random.sum()!=0 else np.arange(5)
                worst_svm = svm_scores['estimator'][svm_scores['test_roc_auc'][estimators].argsort()[order_model]]#[xgboostscores['test_roc_auc'].argmax()]

                # results
                threshold = .5 if mode == 'all' else .5
                results_cv_df = evaluate_classification_wrapper(gi,worst_svm,np.nan_to_num(X_train,nan=-1),y_train,feature_space,groups,mode,dataset='train',threshold = 'optimize',name="svm",save_flag = False)
                RESULTS.append(results_cv_df)
                results_cv_df = evaluate_classification_wrapper(gi,worst_svm,np.nan_to_num(X_test,nan=-1),y_test,feature_space,groups,mode,dataset='test',threshold = results_cv_df['threshold'].values[0],name="svm",save_flag = False)
                RESULTS.append(results_cv_df)
                svm_roc_auc = results_cv_df['roc_auc']['50%']
                
                # results_cv = evaluate_classification_complete(best_svm,np.nan_to_num(X_test),y_test,threshold = threshold,name="svm_cv",save_flag = False)
                # results_cv.pop('conf_matrix')
                # results_cv_df = pd.DataFrame(results_cv, index=[0])
                # results_cv_df['feature_space'] = feature_space
                # results_cv_df['groups'] = "_".join(groups)
                # results_cv_df['mode'] = mode
                # results_cv_df["model"] = "SVM"
                # results_cv_df['n_ratio'] = str(y.sum())+":"+str(len(y)-y.sum())
                

                # make ROC curve for both models
                # Plot the ROC curve
                fpr, tpr, thresholds = roc_curve(y_test, worst_svm.predict_proba(np.nan_to_num(X_test))[:, 1])
                fpr2, tpr2, thresholds = roc_curve(y_test, worst_xgb.predict_proba(X_test)[:, 1])
                # calculate ROC curve
                roc_auc = auc(fpr, tpr)
                roc_auc2 = auc(fpr2, tpr2)
                print('SVM ROC AUCs - overall:',roc_auc,'bootstrap:',svm_roc_auc)
                print('XGB ROC AUCs - overall:',roc_auc2,'bootstrap:',xgb_roc_auc)

                # plt.figure(figsize=(6, 6))
                # plt.plot(fpr, tpr, color='darkorange', lw=2, label='Linear SVM (ROC area={:.2f})'.format(auc(fpr, tpr)))
                # plt.plot(fpr2, tpr2, color='blue', lw=2, label='XGBoost (ROC area={:.2f})'.format(auc(fpr2, tpr2)))
                # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                # # plt.xlim([0.0, 1.0])
                # # plt.ylim([0.0, 1.05])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title(f'ROC_curve_{"".join(groups)}_{feature_space}_{mode}')
                # plt.legend(loc='lower right')
                # plt.savefig(f'figures/ROC_curve_{"".join(groups)}_{feature_space}_{mode}.png')
                roc_aux = pd.DataFrame([['SVM',fpr, tpr,roc_auc,svm_roc_auc],['XGBoost',fpr2, tpr2,roc_auc2,xgb_roc_auc]],columns=['model','fpr','tpr','roc_auc_all','roc_auc_boot'])
                roc_aux['groups'] = "".join(groups)
                roc_aux['feature_space'] = feature_space
                roc_aux['mode'] = mode
                roc_df = pd.concat([roc_df,roc_aux.copy()],axis=0)

    RESULTS = pd.concat(RESULTS,axis=0,ignore_index = False)
    # pdg.to_gbq(RESULTS,''
    # pdg.to_gbq(RESULTS,f"MHC_PH.model_metrics{name_output_file}{external_flag}", project_id=project_id,if_exists="replace")
    RESULTS.to_csv(f"data/model_metrics{name_output_file}{external_flag}.csv",index=True)
    roc_df.to_csv(f"data/roc_auc_df{name_output_file}{external_flag}.csv",index=False)
    print('ml pipeline ran successfully')

def activity_selector(activity_include_now,df_features_all,df_features_all_us):
    if isinstance(activity_include_now,str) and activity_include_now in ['A','B','C','D']:
        query = f"variable in {tuple(allvarplot_dict['Watch'][activity_include_now])}"
        lbl = activity_include_now
    elif isinstance(activity_include_now,dict) and list(activity_include_now.keys())[0] in ('device','feat_ext_type'): 
        key = list(activity_include_now.keys())[0]
        query = f"{key} == '{activity_include_now[key]}'"
        lbl = activity_include_now[key]
    elif isinstance(activity_include_now,str) and activity_include_now in ['A_Watch','A_iPhone','B_Watch','C_Watch','D_Watch']:
        activity,device = activity_include_now.split("_")
        query = f"variable in {tuple(allvarplot_dict['Watch'][activity])} and device == '{device}'"
        lbl = activity_include_now
    elif activity_include_now == None:
        print('activity ignored')
        query = 'patient == patient' # this is a dummy query
    else:
        raise ValueError('activity_include is in the wrong format')

    df_features_allnow = df_features_all.query(query)
    df_features_all_usnow = df_features_all_us.query(query)
    return df_features_allnow,df_features_all_usnow,lbl

keys = pdg.read_gbq("MHC_PH.questionnaire_key_clean_us2",project_id=project_id)

def question_selector(quest_incl,dfq,dfq_us):
    quest_cols_iter = ['patient']+[k for k in list(keys.loc[keys['Sheet'].isin(quest_incl),'Question'].values) if k not in exclusions_questionnaire]
    dfqnow = dfq.loc[:,quest_cols_iter].fillna(-1)
    dfq_usnow = dfq_us.loc[:,quest_cols_iter].fillna(-1)
    return dfqnow,dfq_usnow

def ml_pipeline_quest_wrapper(dfq,df_features_all,dfq_us,df_features_all_us,activity_include = [{'device':'iPhone'}],external_flag='',ml_pipe_flag=True,feature_spaces = ['activity','questionnaire','activity_questionnaire']):
    if ml_pipe_flag:
        # merge features with Sheet
        # activity_include: a list of objects with all possible activity types, e.g. [{'device':'iPhone'},{'feat_ext_type':'ARIMA'},'A','C']
        for activity_include_now in activity_include:
            df_features_allnow,df_features_all_usnow,lbl = activity_selector(activity_include_now,df_features_all,df_features_all_us)
            
            # split by groups of questions
            quest_incls = [['CARDIO DIET SURVEY','Vaping_Smoking','ACTIVITY AND SLEEP SURVEY'],
                        ['SATISFIED SURVEY'],
                        ['PAR-Q QUIZ'],
                        ['Adequacy_of_activity_mindset_me','Exercise_process_mindset_measur','Illness_mindset_inventory'],
                        ['RISK FACTOR SURVEY','DEMOGRAPHICS','DAILY CHECK']]

            for quest_incl in quest_incls:
                dfqnow,dfq_usnow = question_selector(quest_incl,dfq,dfq_us)
                ml_pipeline([dfqnow,df_features_allnow],[dfq_usnow,df_features_all_usnow],ml_pipe_flag=ml_pipe_flag,external_flag='',lbl = lbl+'_'.join(quest_incl),feature_spaces = feature_spaces)
                print(f'question wrapper done! {quest_incl}')
        
            # bonus one
            exclusions_questionnaire_red = ['patient']+[k for k in exclusions_questionnaire if k not in ('heart_disease_Pulmonary_Hypertension','vascular_PAH')]
            dfqnow = dfq.loc[:,exclusions_questionnaire_red].fillna(-1)
            dfq_usnow = dfq_us.loc[:,exclusions_questionnaire_red].fillna(-1)
            ml_pipeline([dfqnow,df_features_all],[dfq_usnow,df_features_all_us],ml_pipe_flag=ml_pipe_flag,external_flag=external_flag,lbl = lbl+'excluded',feature_spaces = feature_spaces)
            print('question wrapper done! excluded')
    else:
        print('Question wrapper did not run')


def ml_pipeline_activity_wrapper(dfq,df_features_all,dfq_us,df_features_all_us,ml_pipe_flag=True,feature_groups=['activity-group-device'],feature_spaces = ['activity','questionnaire','activity_questionnaire']):
    # merge features with Sheet
    # keys = pdg.read_gbq("MHC_PH.questionnaire_key_clean_us2",project_id=project_id)
    
    # quest_cols_iter = ['patient']+[k for k in list(keys.loc[keys['Sheet'].isin(quest_incl),'Question'].values) if k not in exclusions_questionnaire]
    # dfqnow = dfq.loc[:,quest_cols_iter].fillna(-1)
    # dfq_usnow = dfq_us.loc[:,quest_cols_iter].fillna(-1)

    # get combinations of activity feature groups
    if 'feat_ext_type' in feature_groups:
        for feat_ext_type in df_features_all['feat_ext_type'].unique():
            try:
                df_features_allnow = df_features_all.query(f"feat_ext_type == '{feat_ext_type}'")
                df_features_all_usnow = df_features_all_us.query(f"feat_ext_type == '{feat_ext_type}'")

                ml_pipeline([dfq,df_features_allnow],[dfq_us,df_features_all_usnow],ml_pipe_flag=ml_pipe_flag,external_flag='',lbl = feat_ext_type,feature_spaces = feature_spaces)
                print(f'activity by feature extraction types wrapper done! {feat_ext_type} {device}')
            except Exception as e:
                print(e)
    # get combinations of types of devices
    if 'device' in feature_groups:
        for device in ['Watch','iPhone']:
            try:
                df_features_allnow = df_features_all.query(f"device == '{device}'")
                df_features_all_usnow = df_features_all_us.query(f"device == '{device}'")

                ml_pipeline([dfq,df_features_allnow],[dfq_us,df_features_all_usnow],ml_pipe_flag=ml_pipe_flag,external_flag='',lbl = device,feature_spaces = feature_spaces)
                print(f'activity by device wrapper done! {device}')
            except Exception as e:
                print(e)

    if 'activity-group-device' in feature_groups:
        # get combinations of groups of activities
        
        for l,v in allvarplot_dict['Watch'].items():
            for device in ['Watch','iPhone']:
                # df_features_allnow = df_features_all.query(f"variable in {tuple(v.keys())} and device == '{device}' and mode == 'all'")
                # print(device,l,df_features_allnow['patient'].nunique())

                try:
                    df_features_allnow = df_features_all.query(f"variable in {tuple(v.keys())} and device == '{device}'")
                    df_features_all_usnow = df_features_all_us.query(f"variable in {tuple(v.keys())} and device == '{device}'")

                    ml_pipeline([dfq,df_features_allnow],[dfq_us,df_features_all_usnow],ml_pipe_flag=ml_pipe_flag,external_flag='',lbl = l+"_"+device,feature_spaces = feature_spaces)
                    print(f'activity by group of activities & device wrapper done! {l} {device}')
                except Exception as e:
                    print(e)

def calc_data_drift(trainset,testset):
    dfq,df_features_all = trainset
    dfq_us,df_features_all_us = testset

    ####### patient has been removed!!!!!
    # df2 = pd.concat([df2,df2_us.query('patient != "04ee62df-475b-477b-ad74-5a206d8a1946"')],axis=0)
    df_id_mapping = pdg.read_gbq('select patient,`Group` from `imperial-410612.MHC_PH.id_mapping2` where patient != "04ee62df-475b-477b-ad74-5a206d8a1946"',project_id=project_id).drop_duplicates()
    df_id_mapping['Group'] = df_id_mapping['Group'].map({'Not found':'Not found', 'Healthy':'Healthy', 'PAH':'PH', 'COVID':'DC', 'NotPH':'DC', 'PH5':'PH','CV':'DC','CTEPH':'PH'})

    dfq = pd.concat([dfq,dfq_us],axis=0)

    # params = pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/bayesian_hyperparameters.csv')
    res,RESULTS = [],[]
    for mode in ['prediagnosis','postdiagnosis']:
    # for mode in ['prediagnosis']:

        df_features = pd.concat([reformat_features(df_features_all,mode),reformat_features(df_features_all_us,mode)],axis=0)

        for feature_space in ['activity_questionnaire','activity','questionnaire']:
            for gi,groups in enumerate([('PH','Healthy','DC'),('PH','Healthy'),('PH','DC')]):
                print('Analysing...',mode,feature_space,groups)
                if feature_space == 'activity_questionnaire':
                    df_featuresnow = df_features.reset_index().merge(dfq,on='patient',how='outer')
                    df_featuresnow.index = df_featuresnow['patient']
                    df_featuresnow = df_featuresnow.drop(columns=['patient'])

                elif feature_space == 'activity':
                    df_featuresnow = df_features.copy()
                    
                elif feature_space == 'questionnaire':
                    df_featuresnow = dfq.copy()
                    df_featuresnow.index = df_featuresnow['patient']
                    df_featuresnow = df_featuresnow.drop(columns=['patient'])

                # extract x and y
                X = df_featuresnow.replace('nan',np.nan).to_numpy().astype(float)
                # X = df_featuresnow.fillna(-9999).to_numpy()
                # aux = df2.loc[:,["Group","patient"]].drop_duplicates().reset_index(drop=True)
                aux = df_id_mapping.merge(pd.Series(df_featuresnow.index),how="right",on="patient")#.dropna(subset='Group')
                y = (aux["Group"] == "PH").values

                # select only PH & Healthy
                idx = aux["Group"].isin(groups)
                y = y[idx]
                X = X[idx,:]
                
                # calculate metrics for data drift
                idx_uk = aux.loc[idx,"patient"].str.startswith('SPVDU')
                expected,actual = X[idx_uk],X[~idx_uk]
                PSI = calculate_psi(actual, expected)

                aux = pd.DataFrame(PSI,columns=['PSI'])
                aux['mode'] = mode
                aux['groups'] = '_'.join(groups)
                aux['feature_space'] = feature_space
                aux.index = df_featuresnow.columns
                aux2 = aux.groupby(['mode','feature_space','groups']).describe()
                aux2.columns = aux2.columns.map('|'.join).str.strip('|')
                RESULTS.append(aux2.reset_index())
                res.append(aux)
    df = pd.concat(RESULTS,axis=0)
    df_raw = pd.concat(res,axis=0)
    df['significance'] = pd.cut(df['PSI|50%'],bins=[0,.1,.25,np.inf],labels=['insignificant','moderate','significant'])
    
    df.to_csv('data/datadrift.csv',index=False)
    df_raw.to_csv('data/datadriftraw.csv',index=True)


def cohort_comparison(df2,df2_us,plot_flag=True):
    if plot_flag:
        df2['cohort'] = 'UK'
        df2['Group'] = 'UK_'+df2['Group']
        df2_us['cohort'] = 'US'
        df2_us['Group'] = 'US_'+df2_us['Group']        
        df2_ = pd.concat([df2,df2_us],axis=0,ignore_index=True)
        cols_iphone = ['FlightsClimbed', 'StepCount','StepCountPaceMax','StepCountPaceMean', 'FlightsClimbedPaceMax','FlightsClimbedPaceMean']
        df2_ = df2_.query(f'device != "iPhone" or variable in {str(tuple(cols_iphone))}').reset_index(drop=True)

        df2_ = df2_.query('(cohort == "UK" and Group == "PH" and months_diagnosis<1) or cohort == "US" or Group != "PH"')

        df22_,_,_,_ = prepare_data_collated_dist(df2_,[])
# plot it
        for device in ['iPhone','Watch']:
            varsnow = {k: rename_dict[k] for k in list(rename_dict) if rename_dict[k] in df22_.query(f'device == "{device}"')['variable'].unique()}
            for vii,var in enumerate(varsnow.values()):
        # plot the distributions
                xtitlenow = 'Value (min-max scaled)' if vii == len(varsnow.values())-1 else None
                titlenow = 'Value Distributions' if vii == 0 else ''
                
                chart = alt.Chart(df22_).mark_boxplot(opacity=1).encode(
                    y=alt.Y('Group:N',sort=['UK_DC','US_DC','UK_PH','US_PH','UK_Healthy','US_Healthy'], axis=alt.Axis(labels=False)).title(var),#,sort=['DC','PH','Healthy','CV']
                    x=alt.X('value_minmax:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=3)).title(xtitlenow),
                    color=alt.Color('Group:N',scale=alt.Scale(domain=['UK_DC','US_DC','UK_PH','US_PH','UK_Healthy','US_Healthy'],range=['#FFC107','#FFC107',  '#1E88E5', '#1E88E5','#004D40', '#004D40'])),
                    # detail='prediagnosis',
                    tooltip=['variable','Group','value_minmax'],
                    # column='cohort',
                    # row=alt.Row('variable:N', title=None,header=alt.Header(labelFontSize=20),sort=list(varsnow.values()))
                ).transform_filter(alt.FieldEqualPredicate(field='variable', 
                    equal=var)).transform_filter(alt.FieldEqualPredicate(field='device', 
                    equal=device)).properties(title = titlenow,width=500,height=120)
                outchart = chart if vii == 0 else outchart & chart
            outchart.save(f'frontend/www/cohort_comparison_{device}.html')
    else:
        print('cohort comparison ignored!')
