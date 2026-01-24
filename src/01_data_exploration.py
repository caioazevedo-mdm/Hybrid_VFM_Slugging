import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Configuração visual
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


def carregar_e_plotar():
    print("Procurando arquivo .parquet na pasta 'data/raw'...")

    # Agora procuramos por *.parquet
    arquivos = glob.glob(os.path.join("data", "raw", "*.parquet"))

    if not arquivos:
        print("ERRO: Nenhum arquivo .parquet encontrado!")
        print("Baixe o arquivo da pasta 'dataset/4' do GitHub 3W e coloque em 'data/raw'.")
        return

    arquivo = arquivos[0]
    print(f"Lendo arquivo Parquet: {os.path.basename(arquivo)}")

    try:
        # Lê direto em Parquet (muito mais rápido)
        df = pd.read_parquet(arquivo)

        # O Pandas geralmente já reconhece o index se foi salvo como parquet
        # Mas vamos garantir que o timestamp seja o índice
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Plota a Pressão Montante do Choke (P-MON-CKP)
        plt.figure()
        sns.lineplot(data=df, x=df.index, y='P-MON-CKP', color='red', linewidth=1)
        plt.title(f"Pressão no Choke (P-MON-CKP) - {os.path.basename(arquivo)}")
        plt.ylabel("Pressão (Pascal)")
        plt.xlabel("Tempo")

        # Destacar falhas de sensor (onde o valor é NaN)
        # O Parquet lida bem com Nulls, mas é bom checar
        nulos = df['P-MON-CKP'].isnull().sum()
        if nulos > 0:
            print(f"Aviso: Existem {nulos} pontos com falha de sensor (NaN).")

        plt.tight_layout()
        plt.show()

        print("Gráfico gerado com sucesso!")

    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        # Dica de debug se der erro de engine
        print("Dica: Se der erro de 'engine', verifique se instalou: pip install pyarrow")


if __name__ == "__main__":
    carregar_e_plotar()