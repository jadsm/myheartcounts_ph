
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import numpy as np
from utils.constants import project_id
import pandas_gbq as pdg
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def regression_metrics(true_values,predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mse)
    return {"mse":mse,"mae":mae,"r2":r2,"rmse":rmse}

def stepcount_func(x,alpha,beta,gamma):
    
    [Sp,isThrough,Spavg] = x

    # base model
    Sw = beta + alpha * Sp 

    # add extra term if it is a through
    Sw += gamma * Spavg * isThrough.astype(float)

    return Sw

class ImputationWatchPhone():
    def __init__(self,df,window_size = 5,mode = "single_patient",target_variable="StepCount",clean_status = 'clean_1'):
        # parameters
        self.window_size = window_size
        self.mode = mode # "single_patient" or "all_patients"
        self.target_variable = target_variable
        self.clean_status = clean_status
        if target_variable == "StepCount":
            self.func = stepcount_func
        else:
            self.func = lambda x,alpha,beta : alpha + beta*x[2] 
        # df = pd.read_csv("data/all_combined.csv")
        # data
        self.df = df

    def run_data_pipeline(self):
       # filter
        self.filter()
        # reformat
        self.reformat()
        
        # calculate the throughs and means for both imputation and training pipelines
        self.calc_throughs_means(self.data_for_training)
        self.calc_throughs_means(self.data_for_imputing)

        if bool(self.data_for_training):
            self.merge_data("for_training")
        if bool(self.data_for_imputing):
            self.merge_data("for_imputing")
        if bool(self.data_for_keeping):
            self.merge_data("for_keeping")

    def run_train_pipeline(self):
 
        self.traintest_split()

        self.train()

        self.evaluate()

    def run_imputation_pipeline(self):

        X_predict = self.df_for_imputing.loc[:,"iPhone":]
        
        X = self.predict([X_predict[c] for c in X_predict.columns])

        return self.merge_imputed_data(X)
    
    def merge_imputed_data(self,X):
        a = self.df_for_keeping.reset_index()
        a["imputation"] = "kept"
        b = X.reset_index()
        b.rename(columns={"iPhone":"Watch"},inplace=True)
        b["imputation"] = "imputed"
        c = self.df_for_training.reset_index()
        c["imputation"] = "kept"
        df = pd.concat([a,b,c],
                        axis=0)
        df = df[df.columns[[0,1,2,4,3,5,6]]]
        return df.loc[:,:"imputation"].reset_index(drop=True)
        
    def filter(self):        
        # get the step count only
        self.df = self.df.query(f'variable == "{self.target_variable}"').reset_index(drop=True)

        # get only relevant aggregation - this is deprecated since full clean has been implemented
        # self.df = self.df.query('aggregation == "Sum"').reset_index(drop=True)

        # get only clean data
        self.df = self.df.query(f'clean_status == "{self.clean_status}"').reset_index(drop=True)

    def reformat(self):
        # organise by patient 
        L = list(self.df.groupby("patient"))

        # reformat
        self.data_for_training,self.data_for_imputing,self.data_for_keeping = {},{},{}
        variable_count = {'iPhone':0,'Watch':0}

        for l in L:
            aux = l[1].pivot_table(index=['startTime','patient'],columns='device', values='value')
            # if there is only one device - impute directly
            if len(aux.keys()) == 1 and aux.keys()[0] == "iPhone":
                ind = np.ones_like(aux).astype(bool)
                print(l[0],"has only one device:",aux.keys()[0],"for imputing...")
                self.data_for_imputing.update({l[0]:aux[ind]})
            elif len(aux.keys()) == 1 and aux.keys()[0] == "Watch":
                ind = np.ones_like(aux).astype(bool)
                print(l[0],"has only one device (Watch):",aux.keys()[0],"to keep...")
                self.data_for_keeping.update({l[0]:aux[ind]})
            else:
                ind = aux.isna().any(axis=1)
                self.data_for_training.update({l[0]:aux[~ind]}) 
                
                # imputing
                ind = aux['Watch'].isna()
                self.data_for_imputing.update({l[0]:aux[ind]}) 
                
                # keeping
                ind = aux['iPhone'].isna()
                self.data_for_keeping.update({l[0]:aux[ind]}) 

            # count the device incidence - this adds unnecessary complexity
            for k in aux.keys():
                variable_count[k]+=1

            # separate into corresponding & not corresponding data
        print("Device count after reformatting",variable_count)    

    def calc_throughs_means(self,data):
        if data == None:
            data = self.data_for_training
        
        # calculate derivative - for throughs and peaks
        for k in data.keys():
            # if 'iPhone' not in data[k].keys():
            #     print("iPhone not in",k,"skipping...")
            #     continue
            # if 'Watch' not in data[k].keys():
            #     print("Watch not in",k,"skipping...")
            #     continue
            data[k]['through_idx'] = False
            through_idx = argrelextrema(data[k].loc[:,'iPhone'].values, np.less, order=2)[0]
            data[k].iloc[through_idx,-1] = True

                # proof
            # through_idx = argrelextrema(data_for_training[k].loc[:,'iPhone'].values, np.less, order=2)[0]
            # throughs = data_for_training['SPVDU00165'].iloc[through_idx,:]
            # plt.plot(data_for_training['SPVDU00165'].loc[:'2018-08-01','iPhone'])
            # plt.plot(throughs.loc[:'2018-08-01','iPhone'],'r.')
            # plt.show()
            # calculate moving average
            # of observations of specified window size
            data[k]['rolling_means'] = data[k]['iPhone'].rolling(self.window_size,
                                                                closed = 'both',
                                                                min_periods=1,
                                                                center=True).mean()

    def merge_data(self,data_ref):
        # merge all datapoints - ask columns
        exec(f"self.df_{data_ref} = pd.concat(self.data_{data_ref}.values(),axis = 0)")

    def traintest_split(self):
        # separate into training and test sets
        y = self.df_for_training.loc[:,'Watch']
        X = self.df_for_training.loc[:,'iPhone':]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=.3,random_state=123)

    def train(self):
        # OLS algorithm
        res = curve_fit(self.func, [self.X_train[c] for c in self.X_train.columns], self.y_train,
                            full_output=True)#, bounds=(0, [3., 1., 0.5]))

        self.popt, self.pcov, self.full_report, self.mesg, self.ier = res

    def evaluate(self):
        # evaluation
        y_pred = self.func([self.X_test[c] for c in self.X_test.columns],*self.popt)
        self.metrics_test = regression_metrics(self.y_test,y_pred)

        y_pred = self.func([self.X_train[c] for c in self.X_train.columns],*self.popt)
        self.metrics_train = regression_metrics(self.y_train,y_pred)

        print("Train results: ", self.metrics_train)
        print("Test results: ", self.metrics_test)

    def predict(self,X,popt=None):
        if popt is None:
            popt = self.popt
        
        return self.func(X,*popt)

#### this is older code
# def merge_imputation(df,df_imputed,target_variable):
#     # # stitch data together - ignore for now
#     # # select only the relevant data
#     df3 = df.query("device == 'Watch'")
#     df3.loc[:,"imputation"] = "kept"

#     # # merge imputation & clean up
#     aux = df.query("device == 'iPhone'").merge(df_imputed.query("imputation == 'imputed'"),
#                                             on = ["startTime","patient"])
#     aux["device"] = 'Watch'

#     aux = aux.drop(columns=["value"]).rename(columns={"Watch":"value"})
#     df = pd.concat([df.query(f"variable != '{target_variable}' and clean_status == 'clean_1' and device == 'Watch'"),
#                     df3,
#                     aux],axis=0)

#     df["clean_status"] = "imputed"
#     df["imputation"] = df["imputation"].fillna("None")
#     print("finished imputation")
#     df = df.reset_index(drop=True)

#     df['dead'] = df['dead'].astype(str)
#     df['Withdrawn'] = df['Withdrawn'].astype(str)
#     return df
    
def fit_every_patient(df,target_variable='StepCount',gbq_save=False):
    Results = []
    for patient,df_patient in list(df.groupby("patient")):
        try:
        # if True:
            imp2 = ImputationWatchPhone(df_patient,target_variable=target_variable)
            imp2.run_data_pipeline()
            if not bool(imp2.data_for_training):
                print(patient,"no training data, skipping...")   
                continue 
            imp2.run_train_pipeline()
            Results.append({"patient":patient,"popt":imp2.popt, "stdE":np.sqrt(np.diag(imp2.pcov)),
                            "n_train":imp2.X_train.shape[0],
                            "n_test":imp2.X_test.shape[0],
                            "metrics_train":imp2.metrics_train,"metrics_test":imp2.metrics_test
                            })
            print(patient,"trained!")
        except Exception as e:
            print(patient,e)

    Results = pd.DataFrame(Results)
    aux1 = pd.json_normalize(Results["metrics_test"])
    aux2 = pd.json_normalize(Results["metrics_train"])
    new_names_tt,new_names_t = {},{}
    for k in aux1.keys():
        new_names_tt.update({k:k+"_test"})
        new_names_t.update({k:k+"_train"})
    aux1.rename(columns=new_names_tt,inplace=True)
    aux2.rename(columns=new_names_t,inplace=True)
    Results = Results.loc[:,'patient':'n_test']
    Results = pd.concat([Results,aux1,aux2],axis=1)

    s = Results.shape[0]
    # Results["parameter"] = Results.shape[0]*["alpha","beta","gamma"]
    Results = Results.explode(["popt","stdE"])
    Results["parameter"] = s*["alpha","beta","gamma"] if target_variable == "StepCount" else s*["alpha","beta"]
    if gbq_save:
        Results.astype(str).to_gbq("MHC_PH.watchphoneoffset", project_id="imperial-410612",if_exists="replace")
        print("saved to BigQuery")
    return Results

def choose_model(imp,Results,target_variable,save=False):
    thresholds = {'R2':imp.metrics_test['r2'],'overfit':imp.metrics_train['r2']-imp.metrics_test['r2']}
    Results['overfit'] = Results['r2_train']-Results['r2_test']
    print(thresholds) 
    Results['mode'] = 'pooled'
    Results.loc[Results['r2_test']>=thresholds['R2'],'mode'] = 'individual'
    
    # add pooled Results to the Results
    pool_dict = {"patient":'pool',"popt":imp.popt, "stdE":np.sqrt(np.diag(imp.pcov)),
                            "n_train":imp.X_train.shape[0],
                            "n_test":imp.X_test.shape[0]}
    pool_dict['parameter'] = ['alpha','beta','gamma'] if target_variable == 'StepCount' else ['alpha','beta']
    pool_dict.update({k+"_train":v for k,v in imp.metrics_train.items()})
    pool_dict.update({k+"_test":v for k,v in imp.metrics_test.items()})
    pool_df = pd.DataFrame(pool_dict)
    pool_df['overfit'] = pool_df['r2_train']-pool_df['r2_test']
    pool_df['mode'] = 'pooled'

    Results = pd.concat([Results,pool_df],axis=0)
    Results['target_variable'] = target_variable
    
    # save results
    if save:
        Results_out = Results.astype(str)
        pdg.to_gbq(Results_out,
                f"MHC_PH.imputation_fit_results",
                project_id=project_id,
                if_exists="append")
    return Results

def impute_merge_data(df,imp,Results):
    # impute data - pooled
    # select patients
    pooled_patients = [p for p in imp.df_for_imputing.reset_index()['patient'].unique() if p not in Results.loc[Results['mode']=='individual','patient'].unique()]
    idx = [k in list(pooled_patients) for k in imp.df_for_imputing.index.get_level_values('patient')]
    df_imputing = imp.df_for_imputing.loc[idx,:]
    # predict patients
    X_predict = df_imputing.loc[:,"iPhone":]
    Xpool = imp.predict([X_predict[c] for c in X_predict.columns])

    # impute data - individual
    # select patients
    df_imputing = imp.df_for_imputing.loc[~np.array(idx),:]
    # predict patients
    X_predict = df_imputing.loc[:,"iPhone":]
    Xind = []
    for patient in X_predict.index.get_level_values('patient').unique():

        print(patient,'started...')
        popt = Results.query(f'patient == "{patient}"').loc[:,'popt'].values
        X = imp.predict([X_predict.loc[X_predict.index.get_level_values('patient') == patient,c] for c in X_predict.columns],popt)
        Xind.append(X)
        print(patient,'done!')

    # merge data
    X = pd.concat([Xpool]+Xind,axis=0)
    df_imputed = imp.merge_imputed_data(X).rename(columns={'imputed':'value'})
    df.drop(columns=['value'],inplace=True)
    df_new = df_imputed.merge(df,on=['startTime','patient'],how='left')
    df_new['clean_status'] = 'imputed'
    df_new['device'] = 'Watch'
    df_new.rename(columns={'Watch':'value'},inplace=True)
    
    return df_new


  # iPhone variables
  # StepCount                 190705
  # StepCountPaceMean         180323
  # StepCountPaceMax          180322
  # FlightsClimbed            153868
  # FlightsClimbedPaceMean    117570
  # FlightsClimbedPaceMax     117570
  # InBed                      28363
  # BedBound                   28363
  # Awake ratio                 9620
  # DistanceWalkingRunning      4850
  # BasalEnergyBurned           3918
  # AppleStandTime              2884
  # Asleep ratio                1996
  # BodyMass                     258
  # Height                       250
  # bmi                          100
  # HeartRate                     10