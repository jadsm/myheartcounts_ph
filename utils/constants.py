
project_id="imperial-410612"
google_credentials_path = "/Users/juandelgado/Desktop/Juan/code/imperial/creds/google_credentials_mhc.json"

# palette = ["#95E6DF","#e6959c","#4682B4"]# old palette
palette_model = {'XGBoost':"blue","PH":"#1E88E5","Healthy":"#004D40"}#0000CE
palette = {'DC':"#FFC107","PH":"#1E88E5","Healthy":"#004D40"}#0000CE
palette_with_us = {'DC':"#FFC107","PH":"#1E88E5","Healthy":"#004D40",
                   'DC_us':"#FFC107","PH_us":"#1E88E5","Healthy_us":"#004D40"}#0000CE
palette_long = {"PH5":"#1E88E5","PAH":"#1E88E5","COVID":"#FFC107","NotPH":"#FFC107","Healthy":"#004D40",None:'lightgray',
                "PAH_us":"#1E88E5","CV_us":"#FFC107","Healthy_us":"#004D40"}

group_dict = {'PAH':'PH', 'NotPH':'DC', 'PH5':'PH', 'Healthy':'Healthy', 'COVID':'DC', 'PAH':'PH', 'CV':'DC','PAH_us':'PH_us', 'Healthy_us':'Healthy_us',
       'CV_us':'DC_us'}

exclusions_questionnaire = ['medications_to_treat',
                                'prescriptionDrugs',
                                'heart_disease_Angina_heart_chest_pains',
                                'heart_disease_Atrial_fibrillation_Afib',
                                'heart_disease_Congenital_Heart',
                                'heart_disease_Coronary_BlockageStenosis',
                                'heart_disease_Coronary_StentAngioplasty',
                                'heart_disease_Heart_AttackMyocardial_Infarction',
                                'heart_disease_Heart_Bypass_Surgery',
                                'heart_disease_Heart_Failure_or_CHF',
                                'heart_disease_High_Coronary_Calcium_Score',
                                'heart_disease_None_of_the_above',
                                'heart_disease_Pulmonary_Hypertension',
                                'heart_disease_nan',
                                'vascular_PAH',
                                'vascular_Abdominal_Aortic_Aneurysm',
                                'vascular_Carotid_Artery_BlockageStenosis',
                                'vascular_Carotid_Artery_Surgery_or_Stent',
                                'vascular_None_of_the_above',
                                'vascular_Peripheral_Vascular_Disease_BlockageStenosis_Surgery_or_Stent',
                                'vascular_Stroke',
                                'vascular_Transient_Ischemic_Attack_TIA',
                                'vascular_nan',
                                'heartCondition',]# 

agg = {"StepCount":"Sum",
        "FlightsClimbed":"Sum",
        "BasalEnergyBurned":"Sum",
        "AppleStandTime":"Mean",
        'AppleStandHourIdle':"Sum",
        'AppleStandHourStood':"Sum",
        "WalkingHeartRateAverage":"Mean",
        "HeartRate":"Mean",
        "RestingHeartRate":"Mean",
        "HeartRateVariabilitySDNN":"Mean",
        "VO2Max":"Mean",
        "Height":"Mean",
        "BodyMass":"Mean"
        }
to_drop = ("DistanceWalkingRunning",
            "DistanceCycling",
            "RespiratoryRate",
            "ActiveEnergyBurned",
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

devices_to_keep = ('MyHeart','Health','Watch','iPhone')


# this converts from SI units to the most commonly used units
unit_conversion = {"StepCount":1,
                "FlightsClimbed":1,
                "BasalEnergyBurned":1/(1000),# Cal->KCal
                "ActiveEnergyBurned":1/(1000),# Cal->KCal
                "AppleStandTime":1,# mins
                "AppleStandHourIdle":1,# hrs
                "AppleStandHourStood":1,#hrs
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
                'BloodPressureDiastolic':1, # mmHG
                'BloodPressureSystolic':1, # mmHg
                'BodyTemperature':1, # C
                'DietaryCholesterol':1 # mg/dl <100 - good [100-200] - problematic
                }

unit_labels = {"StepCount":'steps',
                "FlightsClimbed":'flights',
                "BasalEnergyBurned":'KCal',# Cal->KCal
                "ActiveEnergyBurned":'KCal',# Cal->KCal
                "AppleStandTime":'min',# mins
                "AppleStandHourIdle":'hrs',# hrs
                "AppleStandHourStood":'hrs',#hrs
                "WalkingHeartRateAverage":'bpm', #bps->bpmin
                'HeartRateReserve':'bpm',
                "HeartRate":'bpm',#bps->bpmin
                "RestingHeartRate":'bpm',#bps->bpmin
                "HeartRateVariabilitySDNN":'ms',
                "VO2Max":'ml/Kg/min',
                "Height":'m',# m
                "BodyMass":'Kg', # Kg,
                "CardiacEffort":'beats/m', 
                "DistanceWalkingRunning":'Km',#Km
                "DistanceCycling":'Km',#Km,
                'RespiratoryRate':'bpm',
                'OxygenSaturation':'%', # spO2 in %
                'BloodPressureDiastolic':'mmHg', # mmHG
                'BloodPressureSystolic':'mmHg', # mmHg
                'BodyTemperature':'celcius', # C
                'DietaryCholesterol':'mg/dl', # mg/dl <100 - good [100-200] - problematic
                'StepCountPaceMean':'steps/min',
                'StepCountPaceMax':'steps/min',
                'FlightsClimbedPaceMean':'flights/min',
                'FlightsClimbedPaceMax':'flights/min',
                'BedBound':'hrs'
                }

rename_dict = {
               'VO2Max':'vo2Max', 'CardiacEffort':'CardiacEffort',
               'BasalEnergyBurned':'basalEnergyBurned','ActiveEnergyBurned':'activeEnergyBurned',
                    
                    'HeartRate':'heartRate','RestingHeartRate':'restingHeartRate',
                     'WalkingHeartRateAverage': 'walkingHeartRate','HeartRateReserve':'HeartRateReserve','HeartRateVariabilitySDNN':'heartRateVariabilitySDNN',

        'Awake':'awake', 'Asleep':'asleep', 
    'BedBound':'InBed>18hr',

    'AppleStandTime':'appleStandTime',
    'StepCount':'stepCount','StepCountPaceMean':'StepCountPaceMean','StepCountPaceMax':'StepCountPaceMax',
    'FlightsClimbed':'flightsClimbed','FlightsClimbedPaceMean':'FlightsClimbedPaceMean', 'FlightsClimbedPaceMax':'FlightsClimbedPaceMax'
    }

import itertools

allvarplot_dict = {'iPhone':{'A':dict(itertools.islice(rename_dict.items(), 13,19))},
                   'Watch':{'A':dict(itertools.islice(rename_dict.items(), 12,19)),# Activity
                            'B':dict(itertools.islice(rename_dict.items(), 4,9)),# HRs
                            'D':dict(itertools.islice(rename_dict.items(), 0,4)),# metabolic
                            'C':dict(itertools.islice(rename_dict.items(), 9,11))}}# sleep

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
                'DietaryCholesterol':50 # mg/dl <100 - good [100-200] - problematic
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
                'DietaryCholesterol':220 # mg/dl <100 - good [100-200] - problematic
                }


