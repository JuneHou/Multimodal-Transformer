# autoreload
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy as np
import pandas as pd

mimic_iv_path = "/data/wang/junh/datasets/physionet.org/files/mimiciv/2.2"
mm_dir = "/data/wang/junh/datasets/multimodal"

output_dir = os.path.join(mm_dir, "preprocessing")
os.makedirs(output_dir, exist_ok=True)

def get_procedures_of_interest(df):
    df = df.copy()

    event_list = ['Foley Catheter', 'PICC Line', 'Intubation', 'Peritoneal Dialysis', 
                            'Bronchoscopy', 'EEG', 'Dialysis - CRRT', 'Dialysis Catheter', 
                            'Chest Tube Removed', 'Hemodialysis']
    event_links_df = pd.DataFrame()
    for event in event_list:
        curr_event_item_id = d_items_df[d_items_df["label"] == event]["itemid"].values[0]

        tmp_dict = {"event": event, "itemid": curr_event_item_id}
        event_links_df = pd.concat([event_links_df, pd.DataFrame(tmp_dict, index=[0])], axis=0, ignore_index=True)

    df = df[df["itemid"].isin(event_links_df['itemid'])]
    df = df.merge(event_links_df, on="itemid", how="left")
    df.drop(columns=["itemid"], inplace=True)
    return df

def get_labs_of_interest(df):
    df = df.copy()

    event_list = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
           'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
           'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
           'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 
                      'Platelet Count', 'Neutrophils', 'Vancomycin'
                  ]

    event_links_df = pd.DataFrame()
    for event in event_list:
        # print(event)
        curr_event_item_id = d_lab_items_df[d_lab_items_df["label"] == event]["itemid"].values[0]

        tmp_dict = {"event": event, "itemid": curr_event_item_id}
        event_links_df = pd.concat([event_links_df, pd.DataFrame(tmp_dict, index=[0])], axis=0, ignore_index=True)

    df = df[df["itemid"].isin(event_links_df['itemid'])]
    df = df.merge(event_links_df, on="itemid", how="left")
    df.drop(columns=["itemid"], inplace=True)

    return df

def get_vitals_of_interest(df):
    df = df.copy()

    event_list = [ #CHART EVENTS
                  'Heart Rate','Non Invasive Blood Pressure systolic',
                    'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 
                    'Respiratory Rate','O2 saturation pulseoxymetry', 
                    'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response']

    event_links_df = pd.DataFrame()
    for event in event_list:
        # print(event)
        curr_event_item_id = d_items_df[d_items_df["label"] == event]["itemid"].values[0]

        tmp_dict = {"event": event, "itemid": curr_event_item_id}
        event_links_df = pd.concat([event_links_df, pd.DataFrame(tmp_dict, index=[0])], axis=0, ignore_index=True)

    df = df[df["itemid"].isin(event_links_df['itemid'])]
    df = df.merge(event_links_df, on="itemid", how="left")
    df.drop(columns=["itemid"], inplace=True)

    rename_dict = {
        'Non Invasive Blood Pressure systolic': 'Systolic BP',
        'Non Invasive Blood Pressure diastolic': 'Diastolic BP',
        'Non Invasive Blood Pressure mean': 'Mean BP',
        'O2 saturation pulseoxymetry': 'O2 Saturation'
    }

    df['event'] = df['event'].replace(rename_dict)
    
    return df

f_path = os.path.join(mimic_iv_path, "hosp", "admissions.csv.gz")
admissions_df = pd.read_csv(f_path, low_memory=False)
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

icustays_df = pd.read_csv(os.path.join(mimic_iv_path, "icu", "icustays.csv.gz"), low_memory=False)
icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])

procedureevents_df = pd.read_csv(os.path.join(mimic_iv_path, "icu", "procedureevents.csv.gz"), low_memory=False)
procedureevents_df['starttime'] = pd.to_datetime(procedureevents_df['starttime'])
procedureevents_df['endtime'] = pd.to_datetime(procedureevents_df['endtime'])
# format='mixed' is not valid, use errors='coerce' instead
procedureevents_df['storetime'] = pd.to_datetime(procedureevents_df['storetime'], errors='coerce')

chartevents_df = pd.read_csv(os.path.join(mimic_iv_path, "icu", "chartevents.csv.gz"), low_memory=False)
chartevents_df['charttime'] = pd.to_datetime(chartevents_df['charttime'])
chartevents_df['storetime'] = pd.to_datetime(chartevents_df['storetime'])

hosp_lab_events = pd.read_csv(os.path.join(mimic_iv_path, "hosp", "labevents.csv.gz"), low_memory=False)
hosp_lab_events['charttime'] = pd.to_datetime(hosp_lab_events['charttime'])
hosp_lab_events['storetime'] = pd.to_datetime(hosp_lab_events['storetime'])

# Drop hosp_lab_events where hadm_id is nan
hosp_lab_events = hosp_lab_events.dropna(subset=['hadm_id'])

d_lab_items_df = pd.read_csv(os.path.join(mimic_iv_path, "hosp", "d_labitems.csv.gz"), low_memory=False)

# Drop rows with missing values
d_lab_items_df = d_lab_items_df.dropna()

# Search labels for something that looks like ph
ph_labels = d_lab_items_df[d_lab_items_df['label'].str.contains('Glucose', case=False)]
print(ph_labels)

d_items_df = pd.read_csv(os.path.join(mimic_iv_path, "icu", "d_items.csv.gz"), low_memory=False)

# procedureevents_df = get_procedures_of_interest(procedureevents_df)
labevents_df = get_labs_of_interest(chartevents_df)
vitals_df = get_vitals_of_interest(chartevents_df)
labevents_df = labevents_df[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'event', 'valuenum']]
vitals_df = vitals_df[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'event', 'valuenum']]

del chartevents_df, hosp_lab_events

# Save the data

from tqdm import tqdm

def calc_time_delta_hrs(icu_intime, charttime):
    return (charttime - icu_intime).total_seconds() / 3600



def add_time_delta(df):
    df = df.copy()

    if 'stay_id' in df.columns:
        stay_id_in_cols = True
    else:
        stay_id_in_cols = False
        df['stay_id'] = None
        
    df['icu_time_delta'] = None
    df['hosp_time_delta'] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if 'charttime' in row:
            ref_time = row['charttime']
        elif 'storetime' in row:
            ref_time = row['storetime']

        curr_admission = admissions_df[(admissions_df['subject_id'] == row['subject_id']) & (admissions_df['hadm_id'] == row['hadm_id'])]

        df.loc[index, 'hosp_time_delta'] = calc_time_delta_hrs(curr_admission['admittime'].iloc[0], ref_time)

        if stay_id_in_cols:
            curr_icu_stay = icustays_df[(icustays_df['subject_id'] == row['subject_id']) & (icustays_df['stay_id'] == row['stay_id'])]
            df.loc[index, 'icu_time_delta'] = calc_time_delta_hrs(curr_icu_stay['intime'].iloc[0], ref_time)
        else:
            curr_pts_icustays = icustays_df[icustays_df['subject_id'] == row['subject_id']]

            for icu_index, icu_row in curr_pts_icustays.iterrows():
                if icu_row['intime'] <= ref_time <= icu_row['outtime']:
                    df.loc[index, 'stay_id'] = icu_row['stay_id']
                    df.loc[index, 'icu_time_delta'] = calc_time_delta_hrs(icu_row['intime'], ref_time)
            

    df = df.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta'])
    return df

def convert_events_table_to_ts_array(df):
    # Ensure 'valuenum' or 'value' columns exist
    value_column = 'valuenum' if 'valuenum' in df.columns else 'value'

    # Create a pivot table
    pivot_df = df.pivot_table(index=['hadm_id', 'hosp_time_delta'], 
                              columns='event', 
                              values=value_column, 
                              aggfunc='first').reset_index()

    # Join with the original DataFrame to get other required columns
    keys = ['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta', 'icu_time_delta']
    merged_df = pd.merge(df[keys].drop_duplicates(), pivot_df, on=['hadm_id', 'hosp_time_delta'])

    # Reorder the columns
    cols = merged_df.columns.tolist()
    cols = [col for col in keys if col in cols] + [col for col in cols if col not in keys]
    merged_df = merged_df[cols]

    # Sort the DataFrame
    merged_df.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta'], inplace=True)

    return merged_df

labevents_df = add_time_delta(labevents_df)
vitals_df = add_time_delta(vitals_df)
concat_df = pd.concat([labevents_df, vitals_df], axis=0, ignore_index=True)

labevents_ts_df = convert_events_table_to_ts_array(labevents_df)
vitals_ts_df = convert_events_table_to_ts_array(vitals_df)

concat_df = convert_events_table_to_ts_array(concat_df)