
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score,roc_auc_score,precision_score,recall_score,roc_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
from utils.constants import *
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

def sensitivity_specificity_calc(y_test, y_prob,threshold=0.5):
        y_pred_threshold = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold,labels=[0,1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity,specificity

def sensitivity_specificity_curve(y_test, y_prob, y_pred,name="bin_class",save_flag = False):
    threshold_values = np.linspace(0, 1, 100)
    sensitivity_values = []
    specificity_values = []

    for threshold in threshold_values:
        sensitivity,specificity = sensitivity_specificity_calc(y_test, y_prob,threshold=threshold)
        sensitivity_values.append(sensitivity)
        specificity_values.append(specificity)
        if save_flag:
            plt.figure(figsize=(8, 6))
            # Plot sensitivity and specificity on the same plot
            plt.plot(1 - np.array(specificity_values), sensitivity_values, marker='o', linestyle='-', color='green', label='Sensitivity/Specificity Curve')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.title('Receiver Operating Characteristic (ROC) Curve with Sensitivity/Specificity Points')
            plt.legend(loc='lower right')
            # plt.show()
    
            plt.savefig(f"figures/sensitivity_specificity_curve_{name}_mar24.png")

def bayesian_search(X,y):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from skopt import BayesSearchCV
    from xgboost import XGBClassifier

    # define search space
    params = dict()
    params['max_depth'] = (3, 5, 8, 12)
    params['n_estimators'] = (5,10)
    params['learning_rate'] = (0.05,0.1,1)
    params['scale_pos_weight'] = (0.5,1,2,3)
    params['reg_alpha'] = (0,1,10)
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=XGBClassifier(objective='binary:logistic', random_state=42,), 
                           search_spaces=params, n_iter=32,
                             cv=cv)
    # perform the search
    search.fit(X, y)
    # report the best result
    print("Best score (Bayesian search):",search.best_score_)
    print("Best parameters (Bayesian search):",search.best_params_)
    return search

def optimize_threshold2(y_test,y_prob,resolution=100):
    thresholds = np.linspace(0.2,.8,resolution,endpoint=False)
    y_test = y_test.astype(int)
    tpr = [np.sum([y_test[ii] and (y_prob[ii] > th).astype(int) for ii in range(y_test.shape[0])]) for th in thresholds]#==
    max_tpr = np.max(tpr)
    ix = np.max(np.where(tpr == max_tpr)[0])
    print('Optimized threshold: Best Threshold =', (thresholds[ix]))
    return thresholds[ix]

def optimize_threshold(y_test,y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    gmeans = np.sqrt(tpr * (1-fpr))
    # gmeans = tpr - fpr
    ix = np.argmax(gmeans)
    print('Optimized threshold: Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    return thresholds[ix]

def bootstrap_auc(model, X, y, n_bootstraps=1000):
  """
  This function calculates the 95% confidence interval of ROC AUC using bootstrapping.

  Args:
      model: The trained binary classification model.
      X: The features data.
      y: The target labels.
      n_bootstraps: The number of bootstrap samples (default: 1000).

  Returns:
      A tuple containing the mean AUC and the 95% confidence interval.
  """
  auc_scores,classib = [],[]
  for _ in range(n_bootstraps):
    # try:
    if True:
        # Resample data with replacement
        idx = resample(range(len(X)), replace=True, n_samples=len(X))
        X_boot = X[idx]
        y_boot = y[idx]

        # Train model on the resampled data
        model.fit(X_boot, y_boot.astype(int))

        # Predict probabilities on the original data
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate AUC score
        # roc_auc = roc_auc_score(y, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        auc_scores.append(roc_auc)
        classib.append(y_boot.sum()/y_boot.shape[0])
    # except:
    #     pass

  # Calculate mean and confidence interval
  mean_auc = np.mean(auc_scores)
  ci_lower = np.percentile(auc_scores, 2.5)
  ci_upper = np.percentile(auc_scores, 97.5)
  n_boot = len(auc_scores)
  classib_mean = np.mean(classib)
  classib_ci_lower = np.percentile(classib, 2.5)
  classib_ci_upper = np.percentile(classib, 97.5)

  return mean_auc, (ci_lower, ci_upper), n_boot,classib_mean,(classib_ci_lower,classib_ci_upper)

def evaluate_classification_complete(xgb_classifier,X_test,y_test,threshold = .5,name="bin_class",n_bootstraps=100,save_flag = False):
    
    y_prob = xgb_classifier.predict_proba(X_test)[:, 1]

    if threshold == 'optimize':
        threshold = optimize_threshold2(y_test,y_prob)

    y_pred = y_prob > threshold
    
    # Evaluate the accuracy of the model - with bootstraps
    bootvars = []
    for _ in range(n_bootstraps):
        try:
            np.random.seed()
            # Resample data with replacement
            idx = resample(range(len(y_test)), replace=True, n_samples=int(len(y_test)*.5))
        
            accuracy = accuracy_score(y_test[idx], y_pred[idx])
            f1 = f1_score(y_test[idx], y_pred[idx])
            precision = precision_score(y_test[idx], y_pred[idx])
            recall = recall_score(y_test[idx], y_pred[idx])
            balanced_accuracy = balanced_accuracy_score(y_test[idx], y_pred[idx],adjusted=True)
            sensitivity,specificity = sensitivity_specificity_calc(y_test[idx], y_prob[idx],threshold=threshold)

            # Plot the ROC curve
            fpr, tpr, thresholds = roc_curve(y_test[idx], xgb_classifier.predict_proba(X_test[idx,:])[:, 1])
            roc_auc = auc(fpr, tpr)
            bootvars.append([accuracy,f1,precision,recall,specificity,roc_auc])
        except Exception as e:
            print(e)
    bootvars = pd.DataFrame(bootvars,columns=['accuracy','f1','precision','recall','specificity','roc_auc'])
    
    # roc_auc2,roc_auc_ci,n_boot,classib_mean,classib_ci = bootstrap_auc(xgb_classifier, X_test, y_test, n_bootstraps=2000)
    
    if save_flag:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"figures/ROC_curve_{name}.png")
    # plt.show()

    # sensitivity - specificity curve
    sensitivity_specificity_curve(y_test, y_prob, y_pred,name,save_flag)

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix using Seaborn
    if save_flag:
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='rocket', annot_kws={"size": 16})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted PH')
        plt.ylabel('True PH')
    # plt.show()
    
        plt.savefig(f"figures/conf_matrix_{name}_mar24.png")

    
    # print(f'ROC AUC ± 95% CI {roc_auc2:.2f}±{roc_auc_ci[0]:.2f}/{roc_auc_ci[1]:.2f} with {n_boot} bootstraps')
    # this is the old dict form
    # results = bootvars.describe().drop(index=['count']).to_dict()
    # results.update({"n_bootstraps":n_bootstraps,
    #                 "conf_matrix":conf_matrix,
    #                 "threshold":threshold})
    
    results = bootvars.describe()
    results["n_bootstraps"] = n_bootstraps
    results["threshold"] = threshold
    # Display the evaluation metrics
    print(results.loc['mean',:])
                
    return results

def regression_metrics(true_values,predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mse)
    return {"mse":mse,"mae":mae,"r2":r2,"rmse":rmse}

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

def evaluate_classification_wrapper(gi,best_model,X_test,y_test,feature_space,groups,mode,dataset='test',threshold = "optimize",name="xgboost",save_flag = False):
    results_cv = evaluate_classification_complete(best_model,X_test,y_test,threshold = threshold,name=f"{name}_cv",save_flag = save_flag)
    if results_cv.loc['mean','precision'] == 0 and gi==2 and feature_space == 'activity':
        results_cv = evaluate_classification_complete(best_model,X_test,y_test,threshold = .5,name=f"{name}_cv",save_flag = save_flag)
    # results_cv.pop('conf_matrix')
    # results_cv_df = pd.DataFrame(results_cv, index=[0])
    results_cv['feature_space'] = feature_space
    results_cv['groups'] = "_".join(groups)
    results_cv['mode'] = mode
    results_cv["model"] = name
    results_cv["dataset"] = dataset
    results_cv['n_ratio'] = str(y_test.sum())+":"+str(len(y_test)-y_test.sum())
    return results_cv

def stat_test_adj(df,df2,group1,group2,covar = ['ethnicity', 'age', 'bmi_mhc', 'dead','gender']):
    try:
    # if True:
        residuals1 = perform_linear_fit(df,group1,covar)
        residuals2 = perform_linear_fit(df2,group2,covar)
        # Perform the Mann-Whitney U test on the residuals
        res_adj_wh = mannwhitneyu(residuals1, residuals2, alternative='two-sided')
        return res_adj_wh.pvalue
    except Exception as e:
        print(e)
        return np.nan

def calculate_all_pvals(pvals_df,var,device,group1, group2,L,lv_idx,name='PH-DC',idxs=[0,1]):
    cols = ["variable",'device','groups',"pvalue","pvalue_adj_wh","pvalue_adj_wh2","pvalue_adj_wh3","pvalue_adj_wh4","pvalue_adj_wh5","pvalue_adj_wh6"]
    res = mannwhitneyu(group1,
                   group2,
                   alternative='two-sided')
    
    myvars = [var,device,name,res.pvalue]
    
    # Perform Mann-Whitney U test with covariate adjustment
    myvars.append(stat_test_adj(L[lv_idx[idxs[0]]][1],L[lv_idx[idxs[1]]][1],group1,group2,covar = ['ethnicity', 'age', 'bmi', 'dead','gender']))

    # Perform Mann-Whitney U test with covariate adjustment - 2 
    myvars.append(stat_test_adj(L[lv_idx[idxs[0]]][1],L[lv_idx[idxs[1]]][1],group1,group2,covar = ['ethnicity', 'age','gender']))

    # Perform Mann-Whitney U test with covariate adjustment - 3
    myvars.append(stat_test_adj(L[lv_idx[idxs[0]]][1],L[lv_idx[idxs[1]]][1],group1,group2,covar = ['ethnicity']))

    # Perform Mann-Whitney U test with covariate adjustment - 4
    myvars.append(stat_test_adj(L[lv_idx[idxs[0]]][1],L[lv_idx[idxs[1]]][1],group1,group2,covar = ['age']))

    # Perform Mann-Whitney U test with covariate adjustment - 5
    myvars.append(stat_test_adj(L[lv_idx[idxs[0]]][1],L[lv_idx[idxs[1]]][1],group1,group2,covar = ['gender']))

    # Perform Mann-Whitney U test with covariate adjustment - 6
    myvars.append(stat_test_adj(L[lv_idx[idxs[0]]][1],L[lv_idx[idxs[1]]][1],group1,group2,covar = ['bmi']))

    aux = pd.DataFrame([myvars],columns = cols)
    pvals_df = pd.concat([pvals_df,aux],axis=0,ignore_index=True)

    # pvals.update({var+"-"+device:res.pvalue})
    # pvals_adj.update({var+"-"+device:stats.false_discovery_control(res.pvalue)})
    # pvals_adj_wh.update({var+"-"+device:myvars[3]})
    return pvals_df

def perform_linear_fit(df,group,covar=['bmi']):
    print(covar)
    # ind = L[0][1]['bmi'].notna()
    # covariate = L[0][1].loc[ind,'bmi']  
    ind = df.loc[:,covar].notna().any(axis=1)
    covariate = df.loc[ind,covar]  
    
    # Fit a linear regression model with the covariate
    X = sm.add_constant(covariate)
    # transform into numeric
    for t in X.keys():
        if X[t].dtype == "object":
            aux = pd.get_dummies(X[t],drop_first=True)
            X = pd.concat([X,aux],axis=1)
            X.drop(columns=[t],inplace=True)
    model = sm.OLS(group.loc[ind], X.fillna(0.).values.astype(float))
    results = model.fit()

    # Get the residuals from the linear regression model
    residuals = results.resid
    return residuals

def reformat_features(df_features_all,mode):
    df_features_all = df_features_all.query(f'mode == "{mode}"').reset_index(drop=True)
    df_features_all["features"] = df_features_all["feat_ext_type"]+"|"+df_features_all["features"]+"|"+df_features_all["variable"]+"|"+df_features_all["device"]
    df_features_all.drop(columns=["variable",'mode','feat_ext_type','device'],inplace=True)
    df_features_all.dropna(subset=['features'],inplace=True)
    return df_features_all.pivot(index="patient",values="value",columns="features")

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       mwburke.github.io.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)