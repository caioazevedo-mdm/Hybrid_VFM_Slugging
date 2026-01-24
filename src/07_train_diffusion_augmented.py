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
# Voltamos para uma rede um pouco maior para lidar com mais dados
LEARNING_RATE = 1e-4  # Diminuí um pouco para estabilizar
BATCH_SIZE = 32  # Aumentei o batch pois agora temos dados
BASE_CHANNELS = 64  # Rede com capacidade média
SEQ_LENGTH = 64  # Tamanho da janela (Pontos que a IA vê)
EPOCHS = 200  # 200 épocas com 6000 dados vale por 5000 épocas antigas
DOWNSAMPLING_RATE = 10  # A cada 10 pontos reais, pegamos 1 (Zoom Out)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']

# --- LIMPEZA ---
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


def carregar_dados_augmentation():
    print("Carregando dados com Janela Deslizante...")
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    path_padrao = os.path.join(diretorio_atual, "data", "raw", "*.parquet")
    path_pai = os.path.join(os.path.dirname(diretorio_atual), "data", "raw", "*.parquet")
    arquivos = glob.glob(path_padrao) + glob.glob(path_pai)

    dfs = []
    # 1. Carrega tudo num tabelão gigante primeiro
    for arq in arquivos:
        try:
            df = pd.read_parquet(arq)
            cols = [c for c in FEATURES if c in df.columns]
            if not cols: continue
            if 'class' in df.columns: df = df[df['class'] != 0]  # Só falhas
            if len(df) > 0: dfs.append(df[cols])
        except:
            pass

    if not dfs: raise ValueError("Sem dados!")

    # Junta todos os arquivos
    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.interpolate(limit_direction='both').fillna(0)
    data_values = df_full.values  # Matriz Numpy

    print(f"Dados Brutos Totais: {len(data_values)} pontos.")

    return data_values


def criar_dataset_inteligente(data_values, scaler):
    # Normaliza TUDO de uma vez
    data_scaled = scaler.fit_transform(data_values)

    sequences = []

    # O tamanho real da janela nos dados originais
    # Se queremos 64 pontos e pulamos de 10 em 10, precisamos de 640 pontos brutos
    janela_bruta = SEQ_LENGTH * DOWNSAMPLING_RATE

    # Loop Deslizante (Data Augmentation)
    # Anda de 5 em 5 passos para criar MUITOS exemplos
    stride = 5

    for i in range(0, len(data_scaled) - janela_bruta, stride):
        # Pega o pedaço bruto (ex: 640 pontos)
        chunk = data_scaled[i: i + janela_bruta]

        # Aplica o Downsampling (Pega 1 a cada 10) -> Vira 64 pontos
        # [::RATE] significa "do começo ao fim, pulando RATE passos"
        chunk_downsampled = chunk[::DOWNSAMPLING_RATE]

        # Garante que tem o tamanho exato (às vezes sobra 1 ponto)
        if len(chunk_downsampled) == SEQ_LENGTH:
            sequences.append(chunk_downsampled)

    return np.array(sequences)


def main():
    # 1. Carregar Dados Brutos
    raw_data = carregar_dados_augmentation()

    # 2. Preparar Scaler e Dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Recortando sequências (Isso aumenta o dataset)...")
    sequences_np = criar_dataset_inteligente(raw_data, scaler)

    # CONVERSÃO PARA TENSOR
    tensor_seq = torch.tensor(sequences_np, dtype=torch.float32).permute(0, 2, 1)

    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"SUCESSO: Dataset agora tem {len(sequences_np)} sequências! (Antes eram 41)")

    # --- CHECK VISUAL RÁPIDO ---
    plt.figure(figsize=(10, 3))
    exemplo = scaler.inverse_transform(sequences_np[0])
    plt.plot(exemplo[:, 0])
    plt.title("Check Visual: A onda ainda existe?")
    plt.show()
    # ---------------------------

    # 3. Modelo (Reforçado)
    block_channels = (BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4)  # 64 -> 128 -> 256
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

    # 4. Treino
    print(f"\nIniciando Treinamento Robusto ({EPOCHS} Épocas)...")
    model.train()
    loss_history = []

    try:
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

            if (epoch + 1) % 10 == 0:  # Printa mais vezes
                print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    except KeyboardInterrupt:
        print("\nParando...")

    # Salvar
    os.makedirs("models_final", exist_ok=True)
    model.save_pretrained("models_final/ddpm_slugging_v3")

    # 5. Geração e Comparação
    print("Gerando dados...")
    model.eval()

    # Gera 2 exemplos para garantir
    noise = torch.randn(2, len(FEATURES), SEQ_LENGTH).to(DEVICE)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    gen_np = noise.permute(0, 2, 1).cpu().numpy()

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Real
    real_sample = scaler.inverse_transform(sequences_np[0])
    axes[0].plot(real_sample[:, 0], color='blue')
    axes[0].set_title("REAL (Zoom Out)")

    # Sintético 1
    syn1 = scaler.inverse_transform(gen_np[0])
    axes[1].plot(syn1[:, 0], color='red', linestyle='--')
    axes[1].set_title("SINTÉTICO Exemplo 1")

    # Sintético 2
    syn2 = scaler.inverse_transform(gen_np[1])
    axes[2].plot(syn2[:, 0], color='orange', linestyle='--')
    axes[2].set_title("SINTÉTICO Exemplo 2")

    plt.tight_layout()
    plt.show()

    # Loss
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.show()


if __name__ == "__main__":
    main()