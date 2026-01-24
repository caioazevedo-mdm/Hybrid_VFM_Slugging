import torch
import pandas as pd
import numpy as np
import glob
import os
from torch.utils.data import DataLoader, TensorDataset
from diffusers import UNet1DModel, DDPMScheduler
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
SEQ_LENGTH = 64
BATCH_SIZE = 16  # Reduzi para garantir estabilidade
EPOCHS = 100  # Aumentei para dar tempo de aprender
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP', 'P-PDG', 'T-PDG', 'QGL']


def carregar_dados_seguro():
    print("Carregando dados de FALHA (Classe 4/1)...")
    arquivos = glob.glob(os.path.join("data", "raw", "*.parquet"))
    dfs = []

    for arq in arquivos:
        try:
            df = pd.read_parquet(arq)
            cols = [c for c in FEATURES if c in df.columns]
            if not cols: continue

            # Filtra falhas
            if 'class' in df.columns:
                df = df[df['class'] != 0]

            if len(df) > 0:
                dfs.append(df[cols])
        except:
            pass

    if not dfs: raise ValueError("Sem dados!")

    df_final = pd.concat(dfs, ignore_index=True)

    # 1. Tratamento de NaNs (Interpolação -> Preenchimento)
    df_final = df_final.interpolate(limit_direction='both').fillna(0)

    # 2. Remover colunas constantes (Variância Zero) -> CAUSA DO NAN
    std = df_final.std()
    cols_uteis = std[std > 1e-6].index.tolist()  # Mantém só o que varia
    df_final = df_final[cols_uteis]

    print(f"Colunas Úteis mantidas: {cols_uteis}")
    print(f"Colunas removidas (constantes): {list(set(FEATURES) - set(cols_uteis))}")
    print(f"Total de pontos: {len(df_final)}")

    return df_final


def treinar_diffusion_v2():
    # 1. Dados
    df = carregar_dados_seguro()

    # Normalização Segura com Scikit-Learn
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df.values)

    # Criar sequências
    sequences = []
    for i in range(0, len(data_scaled) - SEQ_LENGTH, SEQ_LENGTH):
        window = data_scaled[i:i + SEQ_LENGTH]
        sequences.append(window)

    if len(sequences) == 0:
        print("Erro: Poucos dados para criar uma sequência. Baixe mais arquivos!")
        return

    # [Batch, Channels, Time]
    tensor_seq = torch.tensor(np.array(sequences), dtype=torch.float32).permute(0, 2, 1)

    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset pronto: {len(sequences)} sequências.")

    # 2. Modelo
    model = UNet1DModel(
        sample_size=SEQ_LENGTH,
        in_channels=tensor_seq.shape[1],  # Número de colunas dinâmico
        out_channels=tensor_seq.shape[1],
        layers_per_block=2,
        block_out_channels=(32, 64, 128),  # Reduzi o tamanho da rede para dados pequenos
        down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Treino Seguro
    print(f"\nIniciando Treinamento V2 na {DEVICE}...")
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

            # --- O SALVA-VIDAS: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    # Salvar
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/ddpm-vfm-slugging-v2")
    print("\nModelo V2 Salvo!")

    plt.plot(loss_history)
    plt.title("Loss Training V2 (Se cair, funcionou!)")
    plt.show()


if __name__ == "__main__":
    treinar_diffusion_v2()