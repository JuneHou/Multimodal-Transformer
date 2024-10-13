import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    confusion_matrix, roc_auc_score, f1_score, average_precision_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

max_notes = 5
max_len = 128

def calc_avg_cxr_embedding(stays_list):
    # sample_cxr_feats = stays_list[0]['cxr_feats'][0]
    feature_names = ['dn_' + str(i) for i in range(1024)]

    # Create dataframe
    df = pd.DataFrame(columns=feature_names)

    for stay in tqdm(stays_list, total=len(stays_list), desc='Calculating CXR Embeddings'):
        if len(stay['cxr_feats']) == 0:
            curr_emb = np.zeros((1, len(feature_names)))
        else:
            curr_emb = np.array(stay['cxr_feats'])
        curr_emb = curr_emb[max_notes:]
        curr_avg_embeddings = np.mean(curr_emb, axis=0)
        # curr_avg_embeddings = curr_emb[0]
        curr_df = pd.DataFrame(curr_avg_embeddings.reshape(1, -1), columns=feature_names)
        df = pd.concat([df, curr_df], axis=0, ignore_index=True)
    
    return df

def calc_avg_ecg_embedding(stays_list):
    # sample_feats = stays_list[0]['ecg_feats'][0]
    feature_names = ['ecg_' + str(i) for i in range(256)]

    # Create dataframe
    df = pd.DataFrame(columns=feature_names)

    for stay in tqdm(stays_list, total=len(stays_list), desc='Calculating ECG Embeddings'):
        if len(stay['ecg_feats']) == 0:
            curr_emb = np.zeros((1, len(feature_names)))
        else:
            curr_emb = np.array(stay['ecg_feats'])
        # If any entries have NaNs, replace with 0
        curr_emb[np.isnan(curr_emb)] = 0

        # If any entries have inf, replace with 0
        curr_emb[np.isinf(curr_emb)] = 0

        curr_emb[curr_emb > 1e6] = 0
        curr_emb = curr_emb[max_notes:]
        curr_avg_embeddings = np.mean(curr_emb, axis=0)
        # curr_avg_embeddings = curr_emb[0]
        curr_df = pd.DataFrame(curr_avg_embeddings.reshape(1, -1), columns=feature_names)
        df = pd.concat([df, curr_df], axis=0, ignore_index=True)
    
    return df

def calc_avg_text_embedding(stays_list):
    # sample_feats = stays_list[0]['text_embeddings'][0]
    feature_names = ['te_' + str(i) for i in range(768)]

    # Create dataframe
    df = pd.DataFrame(columns=feature_names)

    for stay in tqdm(stays_list, total=len(stays_list), desc='Calculating Text Embeddings'):
        if len(stay['text_embeddings']) == 0:
            curr_emb = np.zeros((1, len(feature_names)))
        else:
            curr_emb = np.array(stay['text_embeddings'])

        # curr_avg_embeddings = curr_emb[0]
        curr_emb = curr_emb[max_notes:]
        curr_avg_embeddings = np.mean(curr_emb, axis=0)
        curr_df = pd.DataFrame(curr_avg_embeddings.reshape(1, -1), columns=feature_names)
        df = pd.concat([df, curr_df], axis=0, ignore_index=True)
    
    return df

def calc_ts_embeddings(stays_list):
    event_list = ['Absolute Neutrophil Count', 'Alkaline Phosphate', 'Anion gap',
        'Bicarbonate', 'Calcium', 'Chloride', 'Creatinine', 'Diastolic BP',
        'GCS - Eye Opening', 'GCS - Motor Response', 'GCS - Verbal Response',
        'Glucose', 'Heart Rate', 'Hematocrit', 'Hemoglobin',
        'Inspired O2 Fraction', 'Magnesium', 'Mean BP', 'O2 Saturation', 'PH',
        'Platelet Count', 'Potassium', 'Respiratory Rate', 'Sodium',
        'Systolic BP', 'Temperature', 'Urea Nitrogen', 'Vancomycin', 'WBC',
        'Weight']

    # Preallocate the DataFrame with expected size and columns for efficiency
    num_rows = len(stays_list)
    column_names = []
    for event in event_list:
        for metric in ['_max', '_min', '_mean', '_variance', '_meandiff', '_meanabsdiff', '_maxdiff', '_sumabsdiff', '_diff', '_npeaks', '_trend']:
            column_names.append(event + metric)
    df = pd.DataFrame(np.nan, index=range(num_rows), columns=column_names)

    for row_index, stay in tqdm(enumerate(stays_list), total=len(stays_list), desc='Calculating Time Series Embeddings'):
        curr_ts = stay['irg_ts']
        
        for i, event in enumerate(event_list[:curr_ts.shape[1]]):
            series = curr_ts[:max_len, i]
            
            if len(series) > 1:
                # Suppress warnings for invalid values and divide by zero
                with np.errstate(invalid='ignore', divide='ignore'):
                    max_val = np.max(series)
                    min_val = np.min(series)
                    mean_val = np.mean(series)
                    var_val = np.var(series)
                    series_diff = np.diff(series)
                    meandiff_val = np.mean(series_diff)
                    meanabsdiff_val = np.mean(np.abs(series_diff))
                    maxdiff_val = np.max(np.abs(series_diff))
                    sumabsdiff_val = np.sum(np.abs(series_diff))
                    diff_val = series[-1] - series[0]
                    peaks, _ = find_peaks(series)
                    npeaks_val = len(peaks)
                    trend_val = np.polyfit(np.arange(len(series)), series, 1)[0] if len(series) > 1 else 0

                # Assign the calculations to the correct place in the DataFrame
                for metric, value in zip(['_max', '_min', '_mean', '_variance', '_meandiff', '_meanabsdiff', '_maxdiff', '_sumabsdiff', '_diff', '_npeaks', '_trend'],
                                         [max_val, min_val, mean_val, var_val, meandiff_val, meanabsdiff_val, maxdiff_val, sumabsdiff_val, diff_val, npeaks_val, trend_val]):
                    df.at[row_index, event + metric] = value
            else:
                # If there is only one value, then we can't calculate any of these
                for metric, value in zip(['_max', '_min', '_mean', '_variance', '_meandiff', '_meanabsdiff', '_maxdiff', '_sumabsdiff', '_diff', '_npeaks', '_trend'],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                    df.at[row_index, event + metric] = value

    return df


def extract_labels(stays_list):
    labels = []
    for stay in stays_list:
        labels.append(stay['label'])
    return np.array(labels)

def evaluate_model(y_true, y_pred, y_pred_prob):
    # Figure out if there are more than 2 classes
    is_single_class = len(y_true.shape) == 1

    if is_single_class:
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        auc =  roc_auc_score(y_true, y_pred_prob)
        auprc = average_precision_score(y_true, y_pred_prob)
        conf_matrix = confusion_matrix(y_true, y_pred)

        prop_outcome = sum(y_true) / len(y_true)

        print(f'Proportion of outcome: {prop_outcome}')
        print(f'F1 Score: {f1}')
        print(f'Accuracy: {acc}')
        print(f'Balanced Accuracy: {balanced_acc}')
        print(f'AUC: {auc}')
        print(f'AUPRC: {auprc}')
        print(f'Confusion Matrix: {conf_matrix}')

        return f1, acc, balanced_acc, auc, conf_matrix

    else:
        # Calculate AUC (micro)
        avg_auc_micro = roc_auc_score(y_true, y_pred, average='micro')
        
        # Calculate AUC (macro)
        avg_auc_macro = roc_auc_score(y_true, y_pred, average='macro')
        
        # Calculate AUC (weighted)
        avg_auc_weighted = roc_auc_score(y_true, y_pred, average='weighted')
        
        # Calculate F1 Score (macro)
        y_pred_binary = np.round(y_pred)
        f1_macro = f1_score(y_true, y_pred_binary, average='macro')

        print(f'AUC (micro): {avg_auc_micro}')
        print(f'AUC (macro): {avg_auc_macro}')
        print(f'AUC (weighted): {avg_auc_weighted}')
        print(f'F1 Score (macro): {f1_macro}')

        return avg_auc_micro, avg_auc_macro, avg_auc_weighted, f1_macro
