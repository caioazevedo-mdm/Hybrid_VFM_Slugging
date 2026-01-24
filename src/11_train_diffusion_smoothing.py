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
from scipy.signal import savgol_filter  # <--- O NOVO AMIGO

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
# 2. CARREGAMENTO COM SMOOTHING
# ==========================================
def carregar_dados_smooth():
    print("Carregando dados e aplicando 'Polimento'...")
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
                # --- APLICANDO FILTRO SAVITZKY-GOLAY (SMOOTHING) ---
                # Isso remove o ruído de alta frequência que confunde a rede
                # window_length=51 (olha 50 vizinhos), polyorder=3 (mantém a curva cúbica)
                for col in cols:
                    # Só aplica se tiver dados suficientes
                    if len(df) > 60:
                        df[col] = savgol_filter(df[col], window_length=51, polyorder=3)

                dfs.append(df[cols])
        except: pass

    if not dfs: raise ValueError("❌ Sem dados!")

    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.interpolate(limit_direction='both').fillna(0)
    return df_full.values

def criar_dataset(raw_values, scaler):
    data_scaled = scaler.fit_transform(raw_values)
    sequences = []
    janela_bruta = SEQ_LENGTH * DOWNSAMPLING_RATE
    stride = 5

    for i in range(0, len(data_scaled) - janela_bruta, stride):
        chunk = data_scaled[i : i + janela_bruta]
        chunk_downsampled = chunk[::DOWNSAMPLING_RATE]
        if len(chunk_downsampled) == SEQ_LENGTH:
            sequences.append(chunk_downsampled)

    return np.array(sequences)

# ==========================================
# 3. TREINAMENTO
# ==========================================
def main():
    # 1. Dados Lisos
    raw_data = carregar_dados_smooth()
    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Criando sequências suavizadas...")
    sequences_np = criar_dataset(raw_data, scaler)
    print(f"Dataset: {len(sequences_np)} sequências limpas.")

    # Check Visual do Smoothing
    plt.figure(figsize=(10, 3))
    plt.plot(raw_data[:200, 0], label="Dado Suavizado (O que a rede vai ver)")
    plt.title("Check de Sanidade: A linha está lisa?")
    plt.legend()
    plt.show()

    tensor_seq = torch.tensor(sequences_np, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Modelo
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

    print(f"\nTreinando V5 (Dados Suavizados)...")
    model.train()
    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
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

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    # 3. Geração
    print("Gerando...")
    model.eval()
    noise = torch.randn(3, len(FEATURES), SEQ_LENGTH).to(DEVICE)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    gen_np = noise.permute(0, 2, 1).cpu().numpy()

    # Desnormaliza
    real_sample_raw = scaler.inverse_transform(sequences_np[0])
    gen_sample_1 = scaler.inverse_transform(gen_np[0])
    gen_sample_2 = scaler.inverse_transform(gen_np[1])

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(real_sample_raw[:, 0], color='blue', linewidth=2)
    axes[0].set_title("REAL (Suavizado)")

    axes[1].plot(gen_sample_1[:, 0], color='red', linestyle='-', linewidth=2)
    axes[1].set_title("SINTÉTICO 1")

    axes[2].plot(gen_sample_2[:, 0], color='orange', linestyle='-', linewidth=2)
    axes[2].set_title("SINTÉTICO 2")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()