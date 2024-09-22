import pandas as pd
import numpy as np


def split_csv_randomly(input_file, n_random=500):
    df = pd.read_csv(input_file)

    total_lines = len(df)

    if total_lines <= n_random:
        raise ValueError("Not enough data.")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df_random = df_shuffled.iloc[:n_random]

    df_rest = df_shuffled.iloc[n_random:]

    output_file_random = "file_random.csv"
    output_file_rest = "file_rest.csv"

    df_random.to_csv(output_file_random, index=False)
    df_rest.to_csv(output_file_rest, index=False)

    print(f"Arquivo '{output_file_random}' criado com {len(df_random)} linhas.")
    print(f"Arquivo '{output_file_rest}' criado com {len(df_rest)} linhas.")


#
split_csv_randomly(input_file="./train.csv", n_random=3000)
