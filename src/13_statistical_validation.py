import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

# Configurações
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']
SEQ_LENGTH = 64
DOWNSAMPLING_RATE = 10


# 1. Carregar Dados Reais
def carregar_dados_rapido():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    path_padrao = os.path.join(diretorio_atual, "data", "raw", "*.parquet")
    path_pai = os.path.join(os.path.dirname(diretorio_atual), "data", "raw", "*.parquet")
    arquivos = glob.glob(path_padrao) + glob.glob(path_pai)

    dfs = []
    for arq in arquivos:
        try:
            df = pd.read_parquet(arq)
            cols = [c for c in FEATURES if c in df.columns]
            if not cols: continue
            if 'class' in df.columns: df = df[df['class'] != 0]
            if len(df) > 0:
                # Filtro Savitzky-Golay na entrada
                for col in cols:
                    if len(df) > 60:
                        df[col] = savgol_filter(df[col], window_length=51, polyorder=3)
                dfs.append(df[cols])
        except:
            pass

    if not dfs: return np.array([])
    df_full = pd.concat(dfs, ignore_index=True)
    return df_full.values


# 2. Processar
print("Carregando dados para validar envelope...")
data = carregar_dados_rapido()

if len(data) == 0:
    print("Erro: Nenhum dado encontrado.")
else:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    sequences = []
    stride = 5
    janela_bruta = SEQ_LENGTH * DOWNSAMPLING_RATE

    for i in range(0, len(data_scaled) - janela_bruta, stride):
        chunk = data_scaled[i: i + janela_bruta]
        chunk_downsampled = chunk[::DOWNSAMPLING_RATE]
        if len(chunk_downsampled) == SEQ_LENGTH:
            sequences.append(chunk_downsampled)

    sequences_np = np.array(sequences)

    # --- CORREÇÃO DO ERRO DE DIMENSÃO ---
    # O scaler espera 2D, mas temos 3D (N, 64, 2).
    # Solução: Achatar -> Transformar -> Desachatar
    N, L, C = sequences_np.shape
    sequences_flat = sequences_np.reshape(-1, C)  # Vira uma lista longa 2D
    real_samples_flat = scaler.inverse_transform(sequences_flat)
    real_samples = real_samples_flat.reshape(N, L, C)  # Volta para 3D

    print(f"Dados prontos: {len(real_samples)} janelas reais analisadas.")

    # 3. O PLOT DA VERDADE
    plt.figure(figsize=(12, 6))

    # Plota até 300 exemplos reais em cinza claro (Nuven de Variação)
    limit = min(len(real_samples), 1000)
    step = 5  # Pula de 5 em 5 para não travar o plot

    for i in range(0, limit, step):
        plt.plot(real_samples[i, :, 0], color='gray', alpha=0.1)

    # Plota um exemplo real forte em Azul
    plt.plot(real_samples[0, :, 0], color='blue', linewidth=2, label='Exemplo Real (Aleatório)')

    # Plota as faixas onde sua IA gerou dados (5.80 a 5.92)
    plt.axhline(y=5.80e6, color='red', linestyle='--', linewidth=2, label='Onde sua IA Gerou (Mín)')
    plt.axhline(y=5.92e6, color='red', linestyle='--', linewidth=2, label='Onde sua IA Gerou (Máx)')

    plt.title("Prova de Conceito: A IA gera dados dentro do envelope real?")
    plt.xlabel("Tempo (Amostras)")
    plt.ylabel("Pressão (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()