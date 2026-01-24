import optuna
import torch
import pandas as pd
import numpy as np
import glob
import os
import gc
from torch.utils.data import DataLoader, TensorDataset
from diffusers import UNet1DModel, DDPMScheduler
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- CONFIG FIXA ---
SEQ_LENGTH = 64
EPOCHS_PER_TRIAL = 30  # Aumentei para dar tempo da rede mostrar serviço
DOWNSAMPLING_RATE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']

# --- LIMPEZA INICIAL ---
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


# --- FUNÇÃO DE CARREGAMENTO (A mesma do Script 7) ---
def carregar_dados_e_processar():
    # 1. Achar arquivos
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

    if not dfs: return None

    # 2. Processar (Sliding Window + Downsampling)
    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.interpolate(limit_direction='both').fillna(0)
    raw_values = df_full.values

    # Usaremos StandardScaler dessa vez (melhor para distribuições normais/ruído)
    scaler = StandardScaler()
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


# Carrega os dados UMA vez na memória para não ficar lento
print("Preparando dados para o Optuna...")
GLOBAL_DATA = carregar_dados_e_processar()
if GLOBAL_DATA is None:
    raise ValueError("❌ Erro ao carregar dados!")
print(f"Dados prontos na memória: {len(GLOBAL_DATA)} sequências.")


# --- O OBJETIVO DO OPTUNA ---
def objective(trial):
    # Limpa memória GPU entre trials
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 1. Hiperparâmetros para Testar
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [32, 64, 128])  # Redes maiores

    # Kernel Size: Tamanho do filtro (Olha 3 pontos vizinhos? ou 5?)
    # Ondas grandes precisam de kernels maiores às vezes
    kernel_size = trial.suggest_categorical("kernel_size", [3])

    # 2. Preparar DataLoader
    tensor_seq = torch.tensor(GLOBAL_DATA, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Modelo Dinâmico
    block_channels = (base_channels, base_channels * 2, base_channels * 4)

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
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 4. Treino Rápido
    model.train()

    for epoch in range(EPOCHS_PER_TRIAL):
        epoch_loss = 0
        steps = 0
        for batch in dataloader:
            clean_images = batch[0].to(DEVICE)
            noise = torch.randn_like(clean_images).to(DEVICE)
            timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=DEVICE).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Se explodir (NaN), corta o trial
            if torch.isnan(loss):
                raise optuna.TrialPruned()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps

        # Reporta e Pruna (Corta testes ruins cedo)
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return avg_loss


if __name__ == "__main__":
    print("Iniciando Optuna Reloaded (Buscando parâmetros para ONDAS)...")

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    # 30 tentativas devem bastar para achar o caminho
    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 40)
    print("NOVO VENCEDOR")
    print(f"Loss Mínima: {study.best_value:.5f}")
    print(f"Parâmetros: {study.best_params}")
    print("=" * 40)