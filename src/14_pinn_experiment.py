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
from scipy.signal import savgol_filter

# ==========================================
# 1. CONFIGURAÇÕES AJUSTADAS
# ==========================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEQ_LENGTH = 64
EPOCHS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']

# --- A CORREÇÃO MÁGICA ---
# Reduzimos o peso drasticamente para equilibrar com o MSE (que é ~0.5)
PHYSICS_WEIGHT = 0.01

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# ==========================================
# 2. CARREGAMENTO
# ==========================================
def carregar_dados_smooth():
    print("Carregando dados...")
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
    stride = 5
    janela_bruta = SEQ_LENGTH * 10

    for i in range(0, len(data_scaled) - janela_bruta, stride):
        chunk = data_scaled[i: i + janela_bruta]
        chunk_downsampled = chunk[::10]
        if len(chunk_downsampled) == SEQ_LENGTH:
            sequences.append(chunk_downsampled)
    return np.array(sequences)


# ==========================================
# 3. FÍSICA ESTABILIZADA
# ==========================================
def calculate_physics_loss(predicted_x0):
    # Trava de segurança: Se a IA alucinar valores gigantes (ex: 1000),
    # trazemos de volta para o limite aceitável (-3 a 3) antes de calcular a física.
    # Isso impede que a Loss exploda para 100.
    predicted_x0_clamped = torch.clamp(predicted_x0, -3.0, 3.0)

    # Derivada Segunda (Aceleração)
    diff1 = predicted_x0_clamped[:, :, 1:] - predicted_x0_clamped[:, :, :-1]
    diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]

    # Penaliza mudanças bruscas
    loss_smoothness = torch.mean(torch.abs(diff2))
    return loss_smoothness


# ==========================================
# 4. TREINAMENTO
# ==========================================
def main():
    raw_data = carregar_dados_smooth()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sequences_np = criar_dataset(raw_data, scaler)

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

    print(f"\nIniciando PINN v2 (Peso Físico: {PHYSICS_WEIGHT})...")
    model.train()

    for epoch in range(EPOCHS):
        epoch_phys_loss = 0
        epoch_mse_loss = 0

        for batch in dataloader:
            clean_images = batch[0].to(DEVICE)
            noise = torch.randn_like(clean_images).to(DEVICE)
            timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=DEVICE).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample

            # 1. Loss MSE (Padrão)
            loss_mse = torch.nn.functional.mse_loss(noise_pred, noise)

            # 2. Estimativa x0 para Física
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(DEVICE)
            alpha_t = alphas_cumprod[timesteps][:, None, None]
            pred_x0 = (noisy_images - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # 3. Loss Física (Agora controlada)
            loss_phys = calculate_physics_loss(pred_x0)

            # Soma ponderada
            total_loss = loss_mse + (PHYSICS_WEIGHT * loss_phys)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_mse_loss += loss_mse.item()
            epoch_phys_loss += loss_phys.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | MSE: {epoch_mse_loss / len(dataloader):.4f} | Physics: {epoch_phys_loss / len(dataloader):.4f}")

    # ==========================================
    # 5. COMPARAÇÃO FINAL
    # ==========================================
    print("Gerando...")
    model.eval()
    noise = torch.randn(1, len(FEATURES), SEQ_LENGTH).to(DEVICE)  # Gera 1 exemplo

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    gen_np = noise.cpu().numpy()

    # Filtro suave leve na saída
    gen_smoothed = np.zeros_like(gen_np)
    for j in range(len(FEATURES)):
        gen_smoothed[0, j, :] = savgol_filter(gen_np[0, j, :], window_length=15, polyorder=3)
    gen_smoothed = np.transpose(gen_smoothed, (0, 2, 1))

    # Plot
    real_sample = scaler.inverse_transform(sequences_np[0])
    gen_sample = scaler.inverse_transform(gen_smoothed[0])

    plt.figure(figsize=(10, 6))
    plt.plot(real_sample[:, 0], color='blue', alpha=0.6, label='Real (Referência)')
    plt.plot(gen_sample[:, 0], color='green', linewidth=2, label='PINN v2 (Equilibrada)')
    plt.title("Resultado Final da PINN: O Equilíbrio entre Física e Dados")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Foto_PINN_v2_FINAL.png")  # Salva direto
    plt.show()


if __name__ == "__main__":
    main()