import torch
import pandas as pd
import numpy as np
import glob
import os
import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from diffusers import UNet1DModel, DDPMScheduler
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter  # <--- Nossa ferramenta de limpeza

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
BASE_CHANNELS = 128
SEQ_LENGTH = 64
EPOCHS = 400
DOWNSAMPLING_RATE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# ==========================================
# 2. CARREGAMENTO (Com Suavização na Entrada)
# ==========================================
def carregar_dados_smooth():
    print("Carregando dados e aplicando 'Polimento' na entrada...")
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
                # Filtro na ENTRADA (Para ensinar física limpa)
                for col in cols:
                    if len(df) > 60:
                        df[col] = savgol_filter(df[col], window_length=51, polyorder=3)
                dfs.append(df[cols])
        except:
            pass

    if not dfs: raise ValueError("Sem dados!")
    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.interpolate(limit_direction='both').fillna(0)
    return df_full.values


def criar_dataset(raw_values, scaler):
    data_scaled = scaler.fit_transform(raw_values)
    sequences = []
    janela_bruta = SEQ_LENGTH * DOWNSAMPLING_RATE
    stride = 5

    for i in range(0, len(data_scaled) - janela_bruta, stride):
        chunk = data_scaled[i: i + janela_bruta]
        chunk_downsampled = chunk[::DOWNSAMPLING_RATE]
        if len(chunk_downsampled) == SEQ_LENGTH:
            sequences.append(chunk_downsampled)
    return np.array(sequences)


# ==========================================
# 3. TREINAMENTO
# ==========================================
def main():
    raw_data = carregar_dados_smooth()
    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Criando sequências...")
    sequences_np = criar_dataset(raw_data, scaler)
    print(f"Dataset: {len(sequences_np)} sequências.")

    tensor_seq = torch.tensor(sequences_np, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet1DModel(
        sample_size=SEQ_LENGTH,
        in_channels=len(FEATURES),
        out_channels=len(FEATURES),
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nIniciando Treinamento ({EPOCHS} Épocas)...")
    model.train()

    # Loop simplificado para focar no resultado final
    for epoch in range(EPOCHS):
        for batch in dataloader:
            clean_images = batch[0].to(DEVICE)
            noise = torch.randn_like(clean_images).to(DEVICE)
            timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=DEVICE).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item():.5f}")

        # ... (Tudo igual até a parte de Gerar) ...

    # ==========================================
    # 4. GERAÇÃO E PÓS-PROCESSAMENTO (CORRIGIDO)
    # ==========================================
    print("Gerando dados e aplicando 'Polimento Final'...")
    model.eval()

    # Gera 3 exemplos
    noise = torch.randn(3, len(FEATURES), SEQ_LENGTH).to(DEVICE)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    # MUDANÇA AQUI: NÃO fazemos permute agora. Mantemos (Batch, Channels, Time)
    # Assim o filtro sabe que a última dimensão é o Tempo (64)
    gen_np = noise.cpu().numpy()

    # --- APLICANDO FILTRO NA SAÍDA ---
    gen_smoothed = np.zeros_like(gen_np)
    for i in range(len(gen_np)):  # Para cada exemplo
        for j in range(len(FEATURES)):  # Para cada sensor (P e T)
            # Agora sim: gen_np[i, j, :] pega os 64 pontos de tempo
            gen_smoothed[i, j, :] = savgol_filter(gen_np[i, j, :], window_length=15, polyorder=3)

    # AGORA sim invertemos para o formato que o Scaler gosta (Time, Channels)
    # De (Batch, Channels, Time) -> (Batch, Time, Channels)
    gen_smoothed = np.transpose(gen_smoothed, (0, 2, 1))

    # Desnormaliza
    real_sample_raw = scaler.inverse_transform(sequences_np[0])
    gen_sample_1 = scaler.inverse_transform(gen_smoothed[0])
    gen_sample_2 = scaler.inverse_transform(gen_smoothed[1])

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Real
    axes[0].plot(real_sample_raw[:, 0], color='blue', linewidth=2, label='Real')
    axes[0].set_title("REAL (Física Pura)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Sintético 1
    axes[1].plot(gen_sample_1[:, 0], color='red', linewidth=2, linestyle='-', label='IA + Filtro')
    axes[1].set_title("SINTÉTICO 1 (Gerado pela IA)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Sintético 2
    axes[2].plot(gen_sample_2[:, 0], color='orange', linewidth=2, linestyle='-', label='IA + Filtro')
    axes[2].set_title("SINTÉTICO 2 (Gerado pela IA)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()