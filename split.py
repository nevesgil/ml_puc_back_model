import pandas as pd
import numpy as np

def split_csv_randomly(input_file, n_random=500):
    # Carrega o arquivo CSV
    df = pd.read_csv(input_file)
    
    # Determina o número total de linhas
    total_lines = len(df)
    
    # Verifica se há linhas suficientes para amostrar
    if total_lines <= n_random:
        raise ValueError("O arquivo CSV contém menos ou igual a 500 linhas. Não é possível fazer uma amostragem de 500 linhas.")

    # Embaralha as linhas do DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Seleciona as linhas aleatórias
    df_random = df_shuffled.iloc[:n_random]
    
    # Seleciona o restante das linhas
    df_rest = df_shuffled.iloc[n_random:]
    
    # Gera os nomes dos arquivos de saída
    output_file_random = 'file_random.csv'
    output_file_rest = 'file_rest.csv'
    
    # Salva os DataFrames em arquivos CSV
    df_random.to_csv(output_file_random, index=False)
    df_rest.to_csv(output_file_rest, index=False)
    
    print(f"Arquivo '{output_file_random}' criado com {len(df_random)} linhas.")
    print(f"Arquivo '{output_file_rest}' criado com {len(df_rest)} linhas.")

# Substitua 'input_file.csv' pelo caminho do seu arquivo CSV
split_csv_randomly(input_file='./train.csv', n_random=3000)