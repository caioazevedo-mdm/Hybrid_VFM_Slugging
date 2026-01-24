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
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PARÂMETROS VENCEDORES (Optuna Reloaded)
# ==========================================
LEARNING_RATE = 0.00026  # Mais lento e preciso
BATCH_SIZE = 32
BASE_CHANNELS = 128  # Rede Gigante (4x maior que antes)
KERNEL_SIZE = 3

# Configurações de Engenharia
SEQ_LENGTH = 64
EPOCHS = 400  # Aumentei para garantir convergência na rede grande
DOWNSAMPLING_RATE = 10  # Zoom Out
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']

# Limpeza de Memória
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# ==========================================
# 2. CARREGAMENTO INTELIGENTE (Sliding Window)
# ==========================================
def carregar_dados_final():
    print("Carregando dados finais...")
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
            if len(df) > 0: dfs.append(df[cols])
        except:
            pass

    if not dfs: raise ValueError("❌ Sem dados!")

    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.interpolate(limit_direction='both').fillna(0)
    return df_full.values


def criar_dataset(raw_values, scaler):
    # Normalização Standard (Média 0, Desvio 1)
    data_scaled = scaler.fit_transform(raw_values)

    sequences = []
    janela_bruta = SEQ_LENGTH * DOWNSAMPLING_RATE
    stride = 5  # Passos curtos para gerar MUITOS dados

    for i in range(0, len(data_scaled) - janela_bruta, stride):
        chunk = data_scaled[i: i + janela_bruta]
        # Aplica o Zoom Out (Downsampling)
        chunk_downsampled = chunk[::DOWNSAMPLING_RATE]

        if len(chunk_downsampled) == SEQ_LENGTH:
            sequences.append(chunk_downsampled)

    return np.array(sequences)


# ==========================================
# 3. TREINAMENTO
# ==========================================
def main():
    # A. Dados
    raw_data = carregar_dados_final()
    scaler = StandardScaler()  # O Optuna preferiu este

    print("Processando sequências...")
    sequences_np = criar_dataset(raw_data, scaler)
    print(f"Dataset Final: {len(sequences_np)} sequências de treino.")

    tensor_seq = torch.tensor(sequences_np, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # B. Modelo (Heavy Weight)
    block_channels = (BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4)  # 128 -> 256 -> 512 canais!

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

    print(f"\nIniciando Treino Pesado ({EPOCHS} Épocas) na {DEVICE}...")
    print("Pode pegar um café, a rede agora é grande!")

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

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")

    # Salvar
    os.makedirs("models_final", exist_ok=True)
    model.save_pretrained("models_final/ddpm_slugging_final_v3")

    # ==========================================
    # 4. GERAÇÃO E PLOTAGEM
    # ==========================================
    print("Gerando dados sintéticos...")
    model.eval()

    # Gera 3 exemplos
    noise = torch.randn(3, len(FEATURES), SEQ_LENGTH).to(DEVICE)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    gen_np = noise.permute(0, 2, 1).cpu().numpy()

    # Desnormalização (Invertendo o StandardScaler)
    real_sample_raw = scaler.inverse_transform(sequences_np[0])
    gen_sample_1 = scaler.inverse_transform(gen_np[0])
    gen_sample_2 = scaler.inverse_transform(gen_np[1])

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Gráfico Real
    axes[0].plot(real_sample_raw[:, 0], color='blue', label='Pressão Real (Pa)')
    axes[0].set_title("REAL (Zoom Out)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfico Sintético 1
    axes[1].plot(gen_sample_1[:, 0], color='red', linestyle='--', label='Sintético')
    axes[1].set_title("IA (Exemplo 1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gráfico Sintético 2
    axes[2].plot(gen_sample_2[:, 0], color='orange', linestyle='--', label='Sintético')
    axes[2].set_title("IA (Exemplo 2)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title("Evolução da Loss (Rede 128 Canais)")
    plt.show()


if __name__ == "__main__":
    main()