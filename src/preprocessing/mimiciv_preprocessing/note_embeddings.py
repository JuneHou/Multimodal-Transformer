import os
import torch
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MIMIC_IV_HAIM_API import split_note_document, get_biobert_embeddings


mm_dir = "/data/wang/junh/datasets/multimodal"
output_dir = os.path.join(mm_dir, "preprocessing")

rad_notes_df = pd.read_pickle(os.path.join(output_dir, "notes_text.pkl"))

icu_rad_notes_df = rad_notes_df[rad_notes_df['stay_id'].notna()]

from tqdm import tqdm
# Set batch size (you can tune this based on your GPU memory)
BATCH_SIZE = 16

# Set device to use GPU with ID 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

icu_rad_notes_df['biobert_embeddings'] = None

def process_batch(chunk_batch, device):
    embeddings_batch = []
    for chunk in chunk_batch:
        curr_embeddings, _ = get_biobert_embeddings(chunk, device)  # Correctly passing the device
        embeddings_batch.append(curr_embeddings.detach().cpu().numpy())
    return embeddings_batch

# Process in batches
for index_start in tqdm(range(0, icu_rad_notes_df.shape[0], BATCH_SIZE)):
    index_end = min(index_start + BATCH_SIZE, icu_rad_notes_df.shape[0])
    batch_df = icu_rad_notes_df.iloc[index_start:index_end]

    for index, row in batch_df.iterrows():
        curr_subject_id = int(row['subject_id'])
        curr_note_id = row['note_id']
        curr_text = row['text']

        chunk_parse, chunk_length = split_note_document(curr_text, 15)

        # Process chunks in the current note in batches
        note_embeddings = []
        for chunk_batch_start in range(0, len(chunk_parse), BATCH_SIZE):
            chunk_batch_end = min(chunk_batch_start + BATCH_SIZE, len(chunk_parse))
            chunk_batch = chunk_parse[chunk_batch_start:chunk_batch_end]
            embeddings_batch = process_batch(chunk_batch, device)
            note_embeddings.extend(embeddings_batch)

        # Store the result in the DataFrame
        icu_rad_notes_df.at[index, 'biobert_embeddings'] = note_embeddings

icu_rad_notes_df.to_pickle(os.path.join(output_dir, "icu_notes_text_embeddings.pkl"))