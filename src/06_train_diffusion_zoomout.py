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

# --- CONFIGURAÇÕES ---
LEARNING_RATE = 9.7e-4
BATCH_SIZE = 16
BASE_CHANNELS = 32
SEQ_LENGTH = 64
EPOCHS = 300  # Reduzi um pouco pois com dados bons ele aprende rápido
DOWNSAMPLING_RATE = 10  # <--- O SEGREDO: Pega 1 ponto a cada 10 (Zoom Out)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']

# --- LIMPEZA ---
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


def carregar_dados_com_zoom_out():
    print("Carregando dados...")
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    # Tenta achar os dados na pasta atual ou acima
    path_padrao = os.path.join(diretorio_atual, "data", "raw", "*.parquet")
    path_pai = os.path.join(os.path.dirname(diretorio_atual), "data", "raw", "*.parquet")
    arquivos = glob.glob(path_padrao) + glob.glob(path_pai)

    dfs = []
    for arq in arquivos:
        try:
            df = pd.read_parquet(arq)
            cols = [c for c in FEATURES if c in df.columns]
            if not cols: continue

            # Filtra apenas Slugging
            if 'class' in df.columns:
                df = df[df['class'] != 0]

            if len(df) > 0:
                # --- APLICANDO DOWNSAMPLING ---
                # Pega linhas alternadas (ex: linha 0, 10, 20...)
                df_resampled = df.iloc[::DOWNSAMPLING_RATE, :]
                dfs.append(df_resampled[cols])
        except:
            pass

    if not dfs: raise ValueError("❌ Sem dados!")

    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.interpolate(limit_direction='both').fillna(0)

    print(f"Dados carregados (com Downsampling {DOWNSAMPLING_RATE}x): {len(df_final)} pontos.")
    return df_final


def main():
    # 1. Carregar
    df_real = carregar_dados_com_zoom_out()

    # 2. Normalizar
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df_real.values)

    # 3. Criar Sequências
    sequences = []
    # Overlap menor para garantir variedade
    step = SEQ_LENGTH // 4
    for i in range(0, len(data_scaled) - SEQ_LENGTH, step):
        sequences.append(data_scaled[i:i + SEQ_LENGTH])

    # --- CHECK DE SANIDADE ---
    # Vamos ver se agora parece uma onda antes de treinar?
    plt.figure(figsize=(10, 3))
    sample_check = scaler.inverse_transform(sequences[0])
    plt.plot(sample_check[:, 0])
    plt.title(f"Check Visual: Isso parece uma onda? (Janela de {SEQ_LENGTH} pontos)")
    plt.show()
    # -----------------------

    tensor_seq = torch.tensor(np.array(sequences), dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset pronto: {len(sequences)} sequências.")

    # 4. Modelo
    block_channels = (BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4)
    model = UNet1DModel(
        sample_size=SEQ_LENGTH,
        in_channels=len(FEATURES),
        out_channels=len(FEATURES),
        layers_per_block=2,
        block_out_channels=block_channels,
        down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Treino
    print(f"\nTreinando V6 com Zoom Out...")
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

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    # 6. Geração Final
    print("Gerando...")
    model.eval()
    noise = torch.randn(1, len(FEATURES), SEQ_LENGTH).to(DEVICE)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    generated_data_real = scaler.inverse_transform(noise.permute(0, 2, 1).cpu().numpy()[0])
    real_data_real = scaler.inverse_transform(sequences[0])

    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(real_data_real[:, 0], label='Real (Zoom Out)', color='blue')
    plt.title("DADO REAL (Agora devemos ver ondas!)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(generated_data_real[:, 0], label='Sintético', color='red', linestyle='--')
    plt.title("DADO SINTÉTICO")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()