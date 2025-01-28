
# Descriptive analsys of MyHeartCounts data for Pulmonary Hypertension (PH)

This repository contains Python files to analyse wearable data for pulmonary hypertension.

**January 2025**

**Publication**: Delgado-SanMartin, J. et al. Remote monitoring of physical activity level and its perception captured by the MyHeart Counts smartphone app identifies pulmonary arterial hypertension. AJRCCM. Submitted Jan 25. Under review.<br> 
**Authors**: Delgado-SanMartin, J. (Imperial College London), Niamh Errington (Imperial College London) <br>
**Dashboard site**: https://mhc-imperial-public.silico.science/


# Main Files

* **pipeline.py**: file to run complete analytics pipeline. Composed of the following functions: 
    - **analyse_prediagnosis** (compute slopes and run an ANOVA test on them - plot distributions), 
    - **plot_raw_data** (plot the data 'as is'), 
    - **compute_walktest** (match walk test data), 
    - **statistical_analysis** (calculate statistics via NLME and statistics), 
    - **cohort_comparison** (compare US and UK cohorts), 
    - **extract_features** (extract features for modelling), 
    - **ml_pipeline_activity_wrapper** (a wrapper to iterate over multiple activity subset selections),
    - **ml_pipeline_quest_wrapper** (a wrapper to iterate over multiple question subset selections), 
    - **activity_selector/question_selector** (select activity or questionnaire data subsets),  
    - **calc_data_drift** (calculate data drift from UK to US), and 
    - **ml_pipeline** (run the ml pipeline).
<br>All functions contain flags to skip the block or a subset thereof.

* **utils**: utilility files: 
    - **constants.py** (constants file),
    - **utils_altair.py** (visualisation support functions), 
    - **utils_ml.py** (machine learning support functions), 
    - **utils_pipeline.py** (main pipeline support functions and classes), and
    - **utils_timeseries.py** (support functions for time series plotting)

* **imperial-mhc-parsers**: submodule of parser utility files to load and transform the raw data as it comes out of MyHeart Counts App / Apple HealthKit.

**Dependencies**
This codebase uses libraries for data manipulation, statistical analysis, and machine learning. The specific libraries can be inspected in **requirements.txt**. The code reads data from a BigQuery instance in Google Cloud. For access to the original data, please contact UK: Prof Allan Lawrie (allan.lawrie@imperial.ac.uk) or US: Anders Johnson (acjohnson@stanford.edu). For BigQuery data schema or website issues please contact Dr Juan Delgado (j.delgado-san-martin@imperial.ac.uk).

# Getting Started

1. Set up virtual environment
2. Install required libraries: `pip install -r requirements.txt`
3. Data should be parsed and in the correct format for the pipeline to work immediately. 
4. Make sure the suffix flag is correct and each individual flag is set to 'True'.
5. Run the pipeline script. Note: the paths may need to be changed.

# Further Notes

* This README provides a general overview of the files based on their naming conventions. Refer to the code itself for specific details about the functionality.
