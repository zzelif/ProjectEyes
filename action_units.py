import os
import pandas as pd
# import numpy as np
# import tensorflow as tf

def extract_au_directory(path, output):
    valid_aus = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    au_columns = [f"AU{str(au).zfill(2)}_r" for au in valid_aus] + [f"AU{str(au).zfill(2)}_c" for au in valid_aus]
    consolidated_data = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)

                # noinspection PyTypeChecker

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()

                if set(au_columns).issubset(df.columns):
                    mean_values = df[au_columns].mean(axis=0).to_dict()
                    image_id = os.path.splitext(file)[0]
                    consolidated_data.append({'image_id': image_id, **mean_values})
                else:
                    print(f"Missing columns for {file}: {set(au_columns) - set(df.columns)}")

    final_df = pd.DataFrame(consolidated_data)
    if not final_df.empty:  # Check if DataFrame is not empty
        final_df.to_csv(output, index=False)
        print(f"Consolidated AUs saved to {output}")
        print(f"Combined {len(final_df)} rows into one dataset.")
    else:
        print("No data was consolidated. Please check your input files.")

def extract_au(path, output):
    valid_aus = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    au_columns = [f"AU{str(au).zfill(2)}_r" for au in valid_aus] + [f"AU{str(au).zfill(2)}_c" for au in valid_aus]
    consolidated_data = []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_path = os.path.join(path, file)

            # noinspection PyTypeChecker

            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            if set(au_columns).issubset(df.columns):
                mean_values = df[au_columns].mean(axis=0).to_dict()
                image_id = os.path.splitext(file)[0]
                consolidated_data.append({'image_id': image_id, **mean_values})
            else:
                print(f"Missing columns for {file}: {set(au_columns) - set(df.columns)}")

    final_df = pd.DataFrame(consolidated_data)
    if not final_df.empty:  # Check if DataFrame is not empty
        final_df.to_csv(output, index=False)
        print(f"Consolidated AUs saved to {output}")
        print(f"Combined {len(final_df)} rows into one dataset.")
    else:
        print("No data was consolidated. Please check your input files.")

# def match_emotions_to_aus(au_csv, relevant_aus, path):
#     df = pd.read_csv(au_csv)
#     results = []
#
#     for _, row in df.iterrows():
#         image_id = row['image_id']
#         matches = {emotion: 0 for emotion in relevant_aus.keys()}
#
#         for emotion, aus in relevant_aus.items():
#             for au in aus:
#                 if row.get(f"AU{au}_r", 0) > 0 or row.get(f"AU{au}_c", 0) > 0:
#                     matches[emotion] += 1
#
#         predicted_emotion = max(matches, key=matches.get)
#         results.append({'image_id': image_id , 'predicted_emotion': predicted_emotion})
#
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(path, index=False)
#     print(f"Predictions saved to {path}")


# csv = "utils/consolidated_au_features.csv"
# specific_aus = {
#         "Angry": [4, 9, 5, 17],
#         "Happy": [6, 7, 10, 12, 14, 20],
#         "Neutral": [2, 5],
#         "Sad": [1, 4, 6, 7, 9, 12, 15, 17, 20],
#         "Surprised": [1, 2, 5, 25, 26]
#     }
# preds_path = "utils/emotion_predictions.csv"
# match_emotions_to_aus(csv, specific_aus, preds_path)

def aggregate_au_realtime(output):
    """
    Aggregates AU features by computing the mean of intensity columns.

    Args:
        output (str): Directory containing OpenFace output CSVs.

    Returns:
        final_df: Singular CSV with the Aggregated AU feature
        dict: Aggregated AU features in a single csv.
    """
    for file in os.listdir(output):
        if file.endswith('.csv'):
            file_path = os.path.join(output, file)
            df = pd.read_csv(file_path)

            au_intensity_columns = [col for col in df.columns if col.endswith('_r')]
            au_occurences_columns = [col for col in df.columns if col.endswith('_c')]
            mean_int_aus = df[au_intensity_columns].mean().to_dict()
            mean_occ_aus = df[au_occurences_columns].mean().round().astype(int).to_dict()

            return mean_int_aus, mean_occ_aus