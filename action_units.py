import os
import pandas as pd

def extract_au(path, output):
    au_columns = [f"AU{i}_r" for i in range(1, 27)] + [f"AU{i}_c" for i in range(1, 27)]
    # relevant_aus = {
    #     "Angry": [4, 9, 5, 17],
    #     "Happy": [6, 7, 10, 12, 14, 20],
    #     "Neutral": [2, 5],
    #     "Sad": [1, 4, 6, 7, 9, 12, 15, 17, 20],
    #     "Surprised": [1, 2, 5, 25, 26]
    # }

    consolidated_data = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)

                # noinspection PyTypeChecker

                df = pd.read_csv(csv_path)
                consolidated_data.append(df)
                # image_id = os.path.splitext(file)[0]
                # if set(au_columns).issubset(df.columns):
                #     consolidated_data.append({'image_id': image_id, **df[au_columns].mean(axis=0)})

    final_df = pd.concat(consolidated_data, ignore_index=True)
    final_df.columns = final_df.columns.str.strip()
    final_df.to_csv(output, index=False)
    print(f"Consolidated AUs saved to {output}")
    print(f"Combined {len(final_df)} rows into one dataset.")