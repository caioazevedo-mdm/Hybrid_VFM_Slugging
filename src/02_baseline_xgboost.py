import pandas as pd
import glob
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO ---
PASTA_RAW = os.path.join("data", "raw")


def carregar_dados():
    print("Carregando dados para treinamento...")

    # Lista todos os parquets
    arquivos = glob.glob(os.path.join(PASTA_RAW, "*.parquet"))

    if len(arquivos) < 2:
        print("ERRO: Preciso de pelo menos 2 arquivos (1 Normal, 1 Falha).")
        print("Baixe um arquivo da Classe 0 (Normal) no GitHub e coloque na pasta.")
        return None

    dfs = []
    for arq in arquivos:
        try:
            df = pd.read_parquet(arq)

            # --- TRUQUE DE ENGENHARIA ---
            # Como saber se é falha ou normal?
            # O Dataset 3W tem uma coluna 'class'.
            # Mas se você baixou arquivos soltos, vamos assumir pelo nome da pasta (ou criar artificialmente para teste)

            # Se a coluna 'class' não existir, vamos criar baseado no arquivo
            # (Assumindo que você baixou um Normal e um Slugging)
            if 'class' not in df.columns:
                # Se o arquivo for da classe 4 (Slugging), rotule como 1. Se for 0, rotule como 0.
                # Como simplificação agora: Se tiver muita variação na pressão, é falha (1).
                # (Isso é provisório só para este teste rápido)
                std_pressure = df['P-MON-CKP'].std()
                label = 1 if std_pressure > 100000 else 0  # Threshold arbitrário para teste
                df['target'] = label
            else:
                # O Dataset 3W usa códigos (0=Normal, 4=Instabilidade). Vamos converter para binário
                # 0 -> 0 (Normal)
                # Qualquer outra coisa -> 1 (Anomalia)
                df['target'] = df['class'].apply(lambda x: 0 if x == 0 else 1)

            dfs.append(df)
            print(f"Carregado: {os.path.basename(arq)} | Linhas: {len(df)} | Label Estimado: {df['target'].iloc[0]}")

        except Exception as e:
            print(f"Erro em {arq}: {e}")

    if not dfs: return None

    # Junta tudo num tabelão só
    df_final = pd.concat(dfs, ignore_index=True)
    return df_final


def treinar_xgboost(df):
    # Seleciona as features (Variáveis de entrada)
    # P-MON-CKP: Pressão Montante Choke
    # T-JUS-CKP: Temperatura Jusante Choke
    # P-PDG: Pressão de Fundo (se tiver)
    features = ['P-MON-CKP', 'T-JUS-CKP', 'P-PDG', 'T-PDG', 'QGL']

    # Filtra colunas que realmente existem no arquivo
    features_reais = [f for f in features if f in df.columns]

    # Preenche falhas de sensor com 0 (Simples, depois melhoramos)
    X = df[features_reais].fillna(0)
    y = df['target']

    print(f"\nTreinando com features: {features_reais}")

    # Separa: 80% para o robô estudar, 20% para a prova final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria o modelo XGBoost (usa GPU se disponível, mas para dataset pequeno CPU é rápida)
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Faz a prova
    y_pred = model.predict(X_test)

    # Mostra o Boletim Escolar
    print("\n" + "=" * 40)
    print("RESULTADO DO BASELINE (XGBoost)")
    print("=" * 40)
    print(classification_report(y_test, y_pred))

    # Plota a Matriz de Confusão (Onde ele errou?)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão (Baseline)")
    plt.xlabel("O Robô previu...")
    plt.ylabel("A Realidade era...")
    plt.show()


if __name__ == "__main__":
    dados = carregar_dados()
    if dados is not None:
        treinar_xgboost(dados)