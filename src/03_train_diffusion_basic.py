import torch
import pandas as pd
import numpy as np
import glob
import os
from torch.utils.data import DataLoader, TensorDataset
from diffusers import UNet1DModel, DDPMScheduler
from torch.optim import AdamW
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
SEQ_LENGTH = 64  # Tamanho da janela de tempo (segundos/pontos)
BATCH_SIZE = 32  # Quantas janelas processar por vez
EPOCHS = 50  # Quantas vezes estudar os dados
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP', 'P-PDG', 'T-PDG', 'QGL']


def carregar_dados_falha():
    print("Carregando apenas dados de FALHA (Classe 4/1)...")
    arquivos = glob.glob(os.path.join("data", "raw", "*.parquet"))
    dfs = []

    for arq in arquivos:
        try:
            df = pd.read_parquet(arq)

            # Garante que temos as colunas necessárias
            cols_existentes = [c for c in FEATURES if c in df.columns]
            if not cols_existentes: continue

            # Filtra apenas linhas de falha (Se tiver coluna class)
            if 'class' in df.columns:
                df = df[df['class'] != 0]  # Pega tudo que NÃO é normal

            if len(df) > 0:
                dfs.append(df[cols_existentes])
        except:
            pass

    if not dfs:
        raise ValueError("Nenhum dado de falha encontrado! Baixe arquivos da Classe 4.")

    df_final = pd.concat(dfs, ignore_index=True)
    # Preenche NaNs com interpolação (suavização) ou 0
    df_final = df_final.interpolate().fillna(0)
    print(f"Total de pontos de falha carregados: {len(df_final)}")
    return df_final, cols_existentes


def criar_sequencias(df, seq_len):
    # Transforma o tabelão comprido em janelas curtas [Batch, Channels, Time]
    data = df.values
    num_samples = len(data) - seq_len

    sequences = []
    for i in range(0, num_samples, seq_len):  # Pulo de seq em seq (sem overlap para economizar)
        window = data[i:i + seq_len]
        sequences.append(window)

    # Converte para Tensor PyTorch e ajusta formato: [Batch, Channels, Time]
    # O Diffusers espera (Batch, Channels, Length)
    tensor_seq = torch.tensor(np.array(sequences), dtype=torch.float32)
    tensor_seq = tensor_seq.permute(0, 2, 1)
    return tensor_seq


def treinar_diffusion():
    # 1. Preparar Dados
    df, cols_usadas = carregar_dados_falha()

    # Normalização Simples (-1 a 1 é ideal para Diffusion)
    data_min = df.min()
    data_max = df.max()
    df_norm = 2 * (df - data_min) / (data_max - data_min) - 1

    dataset = criar_sequencias(df_norm, SEQ_LENGTH)
    dataloader = DataLoader(TensorDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset pronto: {dataset.shape[0]} sequências de tamanho {SEQ_LENGTH}")

    # 2. Criar Modelo UNet 1D
    model = UNet1DModel(
        sample_size=SEQ_LENGTH,
        in_channels=len(cols_usadas),
        out_channels=len(cols_usadas),
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
    ).to(DEVICE)

    # 3. Scheduler (O Maestro do Ruído)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Loop de Treino
    print(f"\nIniciando Treinamento na {DEVICE} por {EPOCHS} épocas...")
    model.train()

    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            clean_images = batch[0].to(DEVICE)

            # Adiciona ruído aleatório
            noise = torch.randn(clean_images.shape).to(DEVICE)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=DEVICE).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Tenta prever o ruído
            noise_pred = model(noisy_images, timesteps).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    # 5. Salvar Modelo
    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/ddpm-vfm-slugging")
    print("\nModelo salvo em 'models/ddpm-vfm-slugging'")

    # Salvar estatísticas de normalização para desfazer depois
    stats = pd.DataFrame({'min': data_min, 'max': data_max})
    stats.to_csv("models/norm_stats.csv")

    # Plota a Loss
    plt.plot(loss_history)
    plt.title("Evolução do Treinamento (Loss)")
    plt.show()


if __name__ == "__main__":
    treinar_diffusion()