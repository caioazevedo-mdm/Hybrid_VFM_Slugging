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

# ==========================================
# 1. CONFIGURAÇÕES VENCEDORAS (Do Optuna)
# ==========================================
# Baseado nos seus resultados: {'lr': 0.00097, 'batch_size': 16, 'base_channels': 32}
LEARNING_RATE = 9.7e-4
BATCH_SIZE = 16
BASE_CHANNELS = 32
SEQ_LENGTH = 64  # Tamanho da janela de tempo
EPOCHS = 500  # Treino longo para alta precisão
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = ['P-MON-CKP', 'T-JUS-CKP']  # Pressão e Temperatura (as colunas úteis)

# ==========================================
# 2. GESTÃO DE MEMÓRIA (Lição do seu Projeto5E)
# ==========================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print(f"⚙Configurando treino na {DEVICE}...")


# ==========================================
# 3. CARREGAMENTO E TRATAMENTO
# ==========================================
def carregar_dados_final():
    print("Carregando dados finais...")
    # Lógica inteligente para achar a pasta 'data' onde quer que ela esteja
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

            # Filtra apenas dados de falha (Slugging) se houver coluna de classe
            if 'class' in df.columns:
                df = df[df['class'] != 0]

            if len(df) > 0: dfs.append(df[cols])
        except:
            pass

    if not dfs:
        raise ValueError("ERRO CRÍTICO: Nenhum dado de falha encontrado. Verifique a pasta 'data/raw'.")

    df_final = pd.concat(dfs, ignore_index=True)
    # Interpolação para remover buracos nos dados (NaNs)
    df_final = df_final.interpolate(limit_direction='both').fillna(0)

    print(f"Dados carregados: {len(df_final)} pontos de falha real.")
    return df_final


# ==========================================
# 4. O MOTOR PRINCIPAL
# ==========================================
def main():
    # --- A. Preparação dos Dados ---
    df_real = carregar_dados_final()

    # Normalização (-1 a 1 é o padrão para Diffusion Models)
    # Guardamos o 'scaler' para desnormalizar no final (Lição do seu Optuna_TCC_v2)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df_real.values)

    # Criação das Sequências (Janelas Deslizantes)
    sequences = []
    step = SEQ_LENGTH // 2  # Overlap de 50%
    for i in range(0, len(data_scaled) - SEQ_LENGTH, step):
        sequences.append(data_scaled[i:i + SEQ_LENGTH])

    tensor_seq = torch.tensor(np.array(sequences), dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_seq)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset formatado: {len(sequences)} sequências de treino.")

    # --- B. Construção do Modelo (UNet) ---
    # Usando a arquitetura leve sugerida pelo Optuna (32 -> 64 -> 128)
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

    # --- C. Loop de Treinamento ---
    print(f"\nIniciando Treinamento Final ({EPOCHS} Épocas)...")
    model.train()
    loss_history = []

    try:
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch in dataloader:
                clean_images = batch[0].to(DEVICE)
                noise = torch.randn_like(clean_images).to(DEVICE)
                timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=DEVICE).long()

                # Forward Process (Adiciona Ruído)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # A Rede tenta prever o ruído
                noise_pred = model(noisy_images, timesteps).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping (Segurança para não explodir)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    except KeyboardInterrupt:
        print("\nTreinamento interrompido manualmente. Salvando o progresso atual...")

    # Salvar o modelo treinado
    os.makedirs("models_final", exist_ok=True)
    model.save_pretrained("models_final/ddpm_slugging_v1")
    print("\nModelo Salvo em 'models_final/ddpm_slugging_v1'")

    # --- D. GERAÇÃO E DESNORMALIZAÇÃO ---
    print("🔮 Gerando dados sintéticos (Inference)...")
    model.eval()

    # Começamos com ruído aleatório puro (distribuição normal)
    noise = torch.randn(1, len(FEATURES), SEQ_LENGTH).to(DEVICE)

    # Processo de Denoising (Reverse Diffusion)
    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            model_output = model(noise, t).sample
            noise = noise_scheduler.step(model_output, t, noise).prev_sample

    # Resultado gerado (ainda normalizado entre -1 e 1)
    generated_data_norm = noise.permute(0, 2, 1).cpu().numpy()[0]

    # DESNORMALIZAÇÃO (Voltando para Pascal e Celsius)
    generated_data_real = scaler.inverse_transform(generated_data_norm)

    # Pega um dado real para comparação
    real_data_norm = sequences[0]
    real_data_real = scaler.inverse_transform(real_data_norm)

    # --- E. PLOTAGEM (Gráficos Profissionais) ---
    plt.figure(figsize=(12, 8))

    # Subplot 1: Real
    plt.subplot(2, 1, 1)
    plt.plot(real_data_real[:, 0], label='Pressão Real (Pa)', color='blue')
    plt.title("DADO REAL (Original Petrobras 3W)")
    plt.ylabel("Pressão (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Sintético
    plt.subplot(2, 1, 2)
    plt.plot(generated_data_real[:, 0], label='Pressão Sintética (Gerada)', color='red', linestyle='--')
    plt.title("DADO SINTÉTICO (Criado pelo Diffusion Model)")
    plt.ylabel("Pressão (Pa)")
    plt.xlabel("Tempo (Passos)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Gráfico de Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title("Evolução do Erro (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()