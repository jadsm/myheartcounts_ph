save_local = False
project_id="imperial-410612"
google_credentials_path = "/Users/juandelgado/Desktop/Juan/code/imperial/creds/google_credentials_mhc.json"

# pathname = "data/raw/UK_Export"
# pathname = "/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/raw"
# zipname = '20240530_UK_Export.zip'
pathname = "/Users/juandelgado/Desktop/Juan/code/imperial/MyHeartCounts/data/US_data"
zipname = 'US_Export.zip'

to_drop = ("DistanceCycling",
            "RespiratoryRate",
            "AppleStandHourStood",
            "AppleStandHourIdle",
            'OxygenSaturation',
            'BloodPressureDiastolic',
            'BloodPressureSystolic',
            'DietaryCholesterol',
            'BodyTemperature')

# type - the bottom ones have been removed except height and weight
# type  - patients
# BodyTemperature              1
# BloodPressureSystolic        3
# BloodPressureDiastolic       4
# DietaryCholesterol           4
# OxygenSaturation             4
# DistanceCycling             16
# RespiratoryRate             18
# Height                      26
# VO2Max                      35
# BodyMass                    36
# AppleStandTime              48
# HeartRateVariabilitySDNN    51
# WalkingHeartRateAverage     52
# RestingHeartRate            52
# FlightsClimbed              53
# ActiveEnergyBurned          54
# BasalEnergyBurned           54
# DistanceWalkingRunning      55
# HeartRate                   56
# StepCount                   57

devices_to_keep = ('Watch','iPhone','MyHeart')#,'Health',
device_mapping = {'':'Watch','MyHeart':'iPhone','iPhone':'iPhone','Watch':'Watch'} # '' corresponds to sleep and MyHeart just has height and weight

# aggregation type - extensive sum over 1 day - intensive mean over 1 day
aggregation_type = {"HKQuantityTypeIdentifierActiveEnergyBurned":"sum",
                    "HKQuantityTypeIdentifierAppleStandTime":"sum",
                    "HKQuantityTypeIdentifierBasalEnergyBurned":"sum",
                    "HKQuantityTypeIdentifierDistanceWalkingRunning":"sum",
                    "HKQuantityTypeIdentifierStepCount":"sum",
                    "HKQuantityTypeIdentifierFlightsClimbed":"sum",
                    "HKQuantityTypeIdentifierStepCountPace":["mean",'max'],
                    "HKQuantityTypeIdentifierFlightsClimbedPace":["mean",'max'],
                    "HKQuantityTypeIdentifierHeartRate":"mean",
                    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":"mean",
                    "HKQuantityTypeIdentifierRestingHeartRate":"mean",
                    "HKQuantityTypeIdentifierVO2Max":"mean",
                    "HKQuantityTypeIdentifierWalkingHeartRateAverage":"mean",
                    'HKQuantityTypeIdentifierBodyMass':'mean',
                    'HKQuantityTypeIdentifierHeight':'mean',
                    'HKQuantityTypeIdentifierDistanceCycling':'sum',
                    'HKQuantityTypeIdentifierOxygenSaturation':'mean',
                    'HKQuantityTypeIdentifierRespiratoryRate':'mean',
                    'HKQuantityTypeIdentifierBloodPressureDiastolic':'mean',
                    'HKQuantityTypeIdentifierBloodPressureSystolic':'mean',
                    'HKWorkoutActivityTypeWalkingtotal_distance':'sum',
                    'HKWorkoutActivityTypeStairClimbingtotal_distance':'sum',
                    'HKWorkoutActivityTypeRunningtotal_distance':'sum',
                    'HKWorkoutActivityTypeWheelchairWalkPacetotal_distance':'mean',
                    'HKWorkoutActivityTypeCyclingtotal_distance':'sum',
                    'HKWorkoutActivityTypeWalkingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeStairClimbingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeRunningenergy_consumed':'sum',
                    'HKWorkoutActivityTypeWheelchairWalkPaceenergy_consumed':'mean',
                    'HKWorkoutActivityTypeCyclingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeMixedCardiototal_distance':'sum',
                    'HKWorkoutActivityTypeSwimmingtotal_distance':'sum',
                    'HKCategoryValueAppleStandHourStood':'sum',
                    'HKWorkoutActivityTypeStairstotal_distance':'sum',
                    'HKWorkoutActivityTypeEllipticaltotal_distance':'sum',
                    'HKWorkoutActivityTypeMixedCardioenergy_consumed':'sum',
                    'HKWorkoutActivityTypeSwimmingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeStairsenergy_consumed':'sum',
                    'HKWorkoutActivityTypeEllipticalenergy_consumed':'sum',
                    'HKCategoryValueSleepAnalysisInBed':'sum',
                    'HKCategoryValueSleepAnalysisAsleep':'sum',
                    'HKCategoryValueSleepAnalysisAwake':'sum',
                    'HKQuantityTypeIdentifierDietaryCholesterol':'mean',
                    'HKQuantityTypeIdentifierBodyTemperature':'mean',
                    'HKQuantityTypeIdentifierDietaryVitaminD':'mean',
                    'HKQuantityTypeIdentifierBloodGlucose':'mean',
                    'HKCategoryValueAppleStandHourIdle':'sum',
                    'HKQuantityTypeIdentifierCardiacEffort':'mean',
                    'HKWorkoutActivityTypeCrossTrainingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeOtherenergy_consumed':'sum',
                    'HKWorkoutActivityTypeHikingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeHighIntensityIntervalTrainingenergy_consumed':'sum',
                    'HKWorkoutActivityTypeTraditionalStrengthTrainingenergy_consumed':'sum',
                    'HKQuantityTypeIdentifierInhalerUsage':'sum',
                    'HKQuantityTypeIdentifierBasalBodyTemperature':'mean',
                    'HKCategoryTypeIdentifierHighHeartRateEvent':'sum',
                    'HKCategoryTypeIdentifierMindfulSession':'sum'}


# this converts from SI units to the most commonly used units
unit_conversion = {"StepCount":1,
                "FlightsClimbed":1,
                "BasalEnergyBurned":1/(1000),# Cal->KCal
                "ActiveEnergyBurned":1/(1000),# Cal->KCal
                "AppleStandTime":1/60,# s->hrs
                "AppleStandHourIdle":1/60,# hrs
                "AppleStandHourStood":1/60,#hrs
                "WalkingHeartRateAverage":60, #bps->bpmin
                "HeartRate":60,#bps->bpmin
                "RestingHeartRate":60,#bps->bpmin
                "HeartRateVariabilitySDNN":1,
                "VO2Max":1,
                "Height":1,# m
                "BodyMass":1, # Kg,
                "DistanceWalkingRunning":1,#Km
                "DistanceCycling":1,#Km,
                'RespiratoryRate':60,
                'OxygenSaturation':1, # spO2 in %
                'BloodPressureDiastolic':1, # mmHg
                'BloodPressureSystolic':1, # mmHg
                'BodyTemperature':1, # C
                'DietaryCholesterol':1, # mg/dl <100 - good [100-200] - problematic
                'InBed':1/3600,#h
                'Asleep':1/3600,#h
                'Awake':1/3600,#h,
                "FlightsClimbedPaceMax":1,
                "FlightsClimbedPaceMean":1,
                "StepCountPaceMax":1,
                "StepCountPaceMean":1,
                "CardiacEffort":1,
                'InhalerUsage':1,
                'BasalBodyTemperature':1,
              'HighHeartRateEvent':1,
              'MindfulSession':1
                }

# variable minimum - in standard
bound_min = {"StepCount":50,
                "FlightsClimbed":0,
                "BasalEnergyBurned":800,# KCal
                "ActiveEnergyBurned":0,# KCal
                "AppleStandTime":0,# s->hrs
                "AppleStandHourIdle":0,# hrs
                "AppleStandHourStood":0,#hrs
                "WalkingHeartRateAverage":50, #bps->bpmin
                "HeartRate":30,#bps->bpmin
                "RestingHeartRate":30,#bps->bpmin
                "HeartRateVariabilitySDNN":0,
                "VO2Max":0,
                "Height":1.4,# m
                "BodyMass":40, # Kg,
                "DistanceWalkingRunning":0,#Km
                "DistanceCycling":0,#Km,
                'RespiratoryRate':10,
                'OxygenSaturation':70, # spO2 in %
                'BloodPressureDiastolic':25, # mmHG
                'BloodPressureSystolic':50, # mmHg
                'BodyTemperature':33, # C
                'DietaryCholesterol':50, # mg/dl <100 - good [100-200] - problematic
                'InBed':0,#h
                'Asleep':0,#h
                'Awake':0,#h
                "FlightsClimbedPaceMax":0,
                "FlightsClimbedPaceMean":0,
                "StepCountPaceMax":0,
                "StepCountPaceMean":0,
                "CardiacEffort":0,
                'InhalerUsage':0,
                'BasalBodyTemperature':0,
              'HighHeartRateEvent':0,
              'MindfulSession':0
                }

bound_max = {"StepCount":100000,
                "FlightsClimbed":500,
                "BasalEnergyBurned":5000,# KCal
                "ActiveEnergyBurned":5000,# KCal
                "AppleStandTime":24,# s->hrs
                "AppleStandHourIdle":24,# hrs
                "AppleStandHourStood":24,#hrs
                "WalkingHeartRateAverage":220, #bps->bpmin
                "HeartRate":220,#bps->bpmin
                "RestingHeartRate":220,#bps->bpmin
                "HeartRateVariabilitySDNN":150,
                "VO2Max":60,
                "Height":2.2,# m
                "BodyMass":200, # Kg,
                "DistanceWalkingRunning":60,#Km
                "DistanceCycling":200,#Km,
                'RespiratoryRate':200,
                'OxygenSaturation':100, # spO2 in %
                'BloodPressureDiastolic':120, # mmHG
                'BloodPressureSystolic':220, # mmHg
                'BodyTemperature':42, # C
                'DietaryCholesterol':220, # mg/dl <100 - good [100-200] - problematic
                'InBed':24,#h
                'Asleep':24,#h
                'Awake':24,#h
                "FlightsClimbedPaceMax":10000,
                "FlightsClimbedPaceMean":10000,
                "StepCountPaceMax":10000,
                "StepCountPaceMean":10000,
                "CardiacEffort":100,
                'InhalerUsage':1000,
                'BasalBodyTemperature':1000,
              'HighHeartRateEvent':1000,
              'MindfulSession':1000
                }

# legacy code
# variable minimum - in standard
# agg = {"StepCount":"Sum",
#         "FlightsClimbed":"Sum",
#         "BasalEnergyBurned":"Sum",
#         "AppleStandTime":"Mean",
#         'AppleStandHourIdle':"Sum",
#         'AppleStandHourStood':"Sum",
#         "WalkingHeartRateAverage":"Mean",
#         "HeartRate":"Mean",
#         "RestingHeartRate":"Mean",
#         "HeartRateVariabilitySDNN":"Mean",
#         "VO2Max":"Mean",
#         "Height":"Mean",
#         "BodyMass":"Mean"
#         }



# Three devices, though "RareDevice" can be erased it was just to check what happen 
# when a weird name of device is used.
# device_type = ['watch','phone'] # other devices may be included

question_cat = {'cannabisSmoking':'symptoms',
                'cannabisVaping':'symptoms',
                'pastVaping':'symptoms',
                'pastSmokeless':'symptoms',
                'currentVaping':'symptoms',
                'currentSmoking':'symptoms',
                'currentSmokeless':'Other',
                'tobaccoProducts':'Other',
                'tobaccoProductsEver':'Other',
                'onsetSmoking':'Other',
                'everQuitSmoking':'Other',
                'pastCannabisSmoking':'symptoms',
                'lastCannabisSmoking':'symptoms',
                'onsetVaping':'Other',
                'everQuitVaping':'Other',
                'durationQuitSmoking':'Other',
                'methodQuitSmoking':'Other',
                'readinessQuitVaping':'Other',
                'durationQuitVaping':'Other',
                'methodQuitVaping':'Other',
                'readinessQuitSmokeless':'Other',
                'everQuitSmokeless':'Other',
                'readinessQuitSmoking':'Other',
                'methodQuitSmokeless':'Other',
                'durationQuitSmokeless':'Other',
                'pastCannabisVaping':'Other',
                'lastCannabisVaping':'Other',
                'feel_worthwhile1':'psyc',
                'feel_worthwhile2':'psyc',
                'feel_worthwhile3':'psyc',
                'feel_worthwhile4':'psyc',
                'riskfactors1':'symptoms',
                'riskfactors2':'symptoms',
                'riskfactors3':'symptoms',
                'riskfactors4':'symptoms',
                'satisfiedwith_life':'psyc',
                'phys_activity':'habits',
                'sleep_diagnosis1':'comorbid',
                'sleep_time':'habits',
                'sleep_time1':'Other',
                'vigorous_act':'habits',
                'sugar_drinks':'habits',
                'alcohol':'habits',
                'fish':'habits',
                'fruit':'habits',
                'grains':'habits',
                'vegetable':'habits',
                "sodium":'habits',
                'labwork':'Other',
                'regionInformation.countryCode':'demo',
                'work':'demo',
                'moderate_act':'habits',
                'atwork':'habits',
                'sleep_diagnosis2':'Other',
                'body_self_healing_in_many_different_circumstances':'psyc',
                'chronic_illness_impact':'psyc',
                'chronic_illness_body_meaning':'psyc',
                'chronic_illness_body_coping':'psyc',
                'chronic_illness_positive_opportunity':'psyc',
                'chronic_illness_management':'psyc',
                'chronic_illness_body_betrayal':'psyc',
                'chronic_illness_more_meaning_in_life':'psyc',
                'chronic_illness_handling':'psyc',
                'body_remarkable_self_healing':'psyc',
                'chronic_illness_spoil':'psyc',
                'chronic_illness_challenge':'psyc',
                'chronic_illness_body_handling':'psyc',
                'chronic_illness_runing_life':'psyc',
                'chronic_illness_body_management':'psyc',
                'chronic_illness_relatively_normal_life':'psyc',
                'chronic_illness_body_failure':'psyc',
                'chronic_illness_empowering':'psyc',
                'body_self_healing_from_most_conditions_and_diseases':'psyc',
                'chronic_illness_body_blame':'psyc',
                'family_history':'symptoms',
                'medications_to_treat':'drugs',
                'heart_disease':'comorbid',
                'vascular':'comorbid',
                'ethnicity':'demo',
                'race':'demo',
                'education':'demo',
                'easy':'psyc',
                'pleasurable':'psyc',
                'relaxing':'psyc',
                'convenient':'psyc',
                'fun':'psyc',
                'social':'psyc',
                'indulgent':'psyc',
                'chestPain':'symptoms',
                'chestPainInLastMonth':'symptoms',
                'dizziness':'symptoms',
                'heartCondition':'comorbid',
                'jointProblem':'comorbid',
                'physicallyCapable':'psyc',
                'prescriptionDrugs':'drugs',
                'unhealthy':'habits',
                'weight':'Other',
                'beneficial':'psyc',
                'disease':'Other',
                'muscles':'Other'}


quest_conversion = {"atwork": {"I spent most of the day walking or using my hands and arms in work that required moderate exertion":2,
                                "I spent most of the day sitting or standing":0,
                                "I spent most of the day lifting or carrying heavy objects or moving most of my body in some other way":3}, 

"sodium":{"I avoid eating prepackaged and processed foods; and I avoid salt when I'm cooking at home":0,
       'I avoid eating prepackaged and processed foods.':2,
       'I avoid eating prepackaged and processed foods; and I avoid eating out, but when I do, I seek out low-sodium options':1,
       "I avoid eating prepackaged and processed foods; and I avoid eating out, but when I do, I seek out low-sodium options;  and I avoid salt when I'm cooking at home":0,
       "I avoid salt when I'm cooking at home":1,
       "I avoid eating out, but when I do, I seek out low-sodium options;  and I avoid salt when I'm cooking at home":0,
       'I avoid eating out, but when I do, I seek out low-sodium options.':1},

 "beneficial": {"Moderately beneficial for my health":1,
				  "Very beneficial for my health":2,
				  "Slightly harmful for my health":-1,
				  "Moderately harmful for my health":-2,
				  "Neither harmful nor beneficial for my health":0,
				  "Extremely beneficial for my health":3,
				  "Very harmful for my health":-3},  

 
 "convenient": {"Somewhat convenient":1,
			  "Somewhat inconvenient":-1,
			  "Very convenient":2,
			  "Very inconvenient":-2},  


 "disease": {"Decreaess my risk slightly":1,
			  "Decreaess my risk moderately":2,
			  "Neither increases nor decreases my risk":0,
			  "Increases my risk moderately":-2,
			  "Increases my risk slightly":-1,
			  "Decreaess my risk very much":3,
			  "Increases my risk very much":-3},  


 "easy": {"Somewhat difficult":-1,
		  "Very easy":2,
		  "Very difficult":-1,
		  "Somewhat easy":1},  
 
 "fun": {"Somewhat fun":1,
		  "Very fun":2,
		  "Somewhat boring":-1,
		  "Very boring":-2},  


 "indulgent": {"Somewhat indulgent":1,
			  "Somewhat depriving":-1,
			  "Very indulgent":2,
			  "Very depriving":-2},  


 "muscles": {"Strengthening slightly":1,
			  "Strengthening moderately":2,
			  "Weakening slightly":-1,
			  "Neither strengthening nor weakening":0,
			  "Strengthening very much":3,
			  "Weakening moderately":-2,
			  "Weakening very much":-3},  

 "phys_activity": {"About three times a week, I did moderate activities":2,
				  "Once or twice a week, I did light activities":1,
				  "About three times a week, I did vigorous activities":3,
				  "I did not do much physical activity":0,
				  "Almost daily, that is five or more times a week, I did moderate activities":4,
				  "Almost daily, that is, five or more times a week, I did vigorous activities":5},  

 "pleasurable": {"Somewhat pleasurable":1,
				  "Very pleasurable":2,
				  "Very unpleasant":-2,
				  "Somewhat unpleasant":-1},  


 "relaxing": {"Somewhat relaxing":1,
			  "Very relaxing":2,
			  "Somewhat stressful":-1,
			  "Very stressful":-2},  

 "riskfactors1": {"A little":1,
				  "Moderately":2,
				  "Not at all":0,
				  "A lot":3,
				  "Extremely":4},  

 "riskfactors2": {"Higher than average":3,
				  "Much lower than average":0,
				  "Average":2,
				  "Lower than average":1,
				  "Much higher than average":4},  

 "riskfactors3": {"Moderately":2,
				  "A lot":3,
				  "A little":1,
				  "Extremely":4,
				  "Not at all":0},  

 "riskfactors4": {"Higher than average":3,
				  "Lower than average":1,
				  "Average":2,
				  "Much lower than average":0,
				  "Much higher than average":4},  

 "social": {"Somewhat social":1,
				  "Very lonely":-2,
				  "Somewhat lonely":-1,
				  "Very social":2}
				  }

agg_map =  {"Strongly Disagree":-3,
            "Disagree":-2,
            "Somewhat Disagree":-1,
            "Somewhat Agree":1,
            "Agree":2,
            "Strongly Agree":3}