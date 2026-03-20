"""
=============================================================
  ETAPA 3 — MODELO DE LOTAÇÃO DE AEROPORTO
  Arquitetura: LSTM bidirecional + camada de atenção temporal
  O que faz: aprende padrões históricos de tráfego por aeroporto
  e prevê quantos voos e passageiros haverá nas próximas 12 horas
=============================================================
"""

import gc                                        # módulo de garbage collection — libera RAM manualmente
import torch                                     # biblioteca principal de deep learning
import torch.nn as nn                            # submódulo com camadas neurais (LSTM, Linear, etc.)
import numpy as np                               # operações matemáticas com arrays
import polars as pl                              # leitura de Parquet e manipulação de dados (substitui Pandas para grandes volumes)
from torch.utils.data import Dataset, DataLoader # Dataset = estrutura de dados para treino; DataLoader = carrega em batches
from sklearn.preprocessing import StandardScaler # normaliza os dados: média 0, desvio 1
from pathlib import Path                         # manipulação de caminhos de arquivos
import logging                                   # registro de progresso no terminal

log = logging.getLogger(__name__)                # cria um logger com o nome deste arquivo
logging.basicConfig(                             # configura formato e nível do log
    level=logging.INFO,                          # mostra mensagens de INFO para cima
    format="%(asctime)s  %(levelname)s  %(message)s"  # formato: hora + nível + mensagem
)

FEATURES_DIR = Path("data/features")            # pasta onde estão os parquets de features gerados pelo script 02
MODELS_DIR   = Path("models")                   # pasta onde o modelo treinado será salvo
MODELS_DIR.mkdir(exist_ok=True)                 # cria a pasta se não existir (sem erro se já existir)

# ── dicionário de hiperparâmetros do modelo ─────────────────────────────────
CONFIG = {
    "janela_historica": 48,   # quantas horas do passado o modelo "vê" para fazer a previsão (48h = 2 dias)
    "horizonte":        12,   # quantas horas no futuro o modelo prevê (12h à frente)
    "hidden_size":      64,   # tamanho do vetor de estado interno do LSTM (mais = mais capacidade, mais RAM)
    "num_layers":       2,    # número de LSTMs empilhados (stacked LSTM — o 2º processa a saída do 1º)
    "dropout":          0.2,  # 20% dos neurônios são desligados aleatoriamente durante o treino (evita overfitting)
    "batch_size":       32,   # quantos exemplos são processados de uma vez por passagem
    "epochs":           20,   # quantas vezes o modelo vê todo o dataset de treino
    "lr":               1e-3, # taxa de aprendizado: quão grande é o passo de atualização dos pesos (0.001)
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # usa GPU se disponível, senão CPU
}

# ── colunas que entram no modelo como série temporal ─────────────────────────
FEATURES_SERIE = [
    "total_decolagens",      # quantos voos decolaram naquela hora naquele aeroporto (target 1)
    "total_passageiros",     # quantos passageiros embarcaram (target 2)
    "total_assentos",        # capacidade total ofertada naquela hora
    "ocupacao_media",        # taxa média de ocupação dos voos daquela hora
    "voos_distintos",        # quantas companhias diferentes operaram naquela hora
    "dia_semana_num",        # dia da semana como número (0=segunda ... 6=domingo) — captura padrão semanal
    "nr_hora_partida_real",  # hora do dia (0–23) — captura padrão intradiário (pico manhã/tarde)
    "semana_ano",            # semana do ano (1–52) — captura sazonalidade anual (férias, feriados)
    "flag_feriado",          # 1 se é feriado nacional, 0 caso contrário — captura spikes de demanda
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET — converte séries temporais em janelas de (X, y)
# ══════════════════════════════════════════════════════════════════════════════

class AeroportoDataset(Dataset):
    """
    Transforma a série temporal de cada aeroporto em exemplos de treino.
    Cada exemplo = uma janela de 48h de histórico (X) + as próximas 12h (y).
    Usa lazy loading: guarda só os índices na RAM, não os dados em si.
    """

    def __init__(self, df: pl.DataFrame, scaler: StandardScaler):
        self.janela    = CONFIG["janela_historica"]  # tamanho da janela de entrada (48h)
        self.horizonte = CONFIG["horizonte"]          # tamanho do horizonte de previsão (12h)
        self.series    = []                           # lista que vai guardar as séries normalizadas por aeroporto

        for icao in df["sg_icao_origem"].unique().to_list():  # itera sobre cada aeroporto único
            serie = (
                df.filter(pl.col("sg_icao_origem") == icao)         # filtra apenas as linhas deste aeroporto
                  .sort(["dt_partida_real", "nr_hora_partida_real"]) # ordena por data e hora (essencial para séries temporais)
                  .select(FEATURES_SERIE)                            # mantém só as colunas de features
                  .to_numpy().astype(np.float32)                     # converte para array numpy de float32
            )
            if len(serie) >= self.janela + self.horizonte:           # só usa o aeroporto se tiver dados suficientes
                self.series.append(scaler.transform(serie))          # normaliza a série e adiciona à lista

        # guarda apenas os índices, não os tensores pré-computados
        # isso economiza muita RAM — os tensores são criados só quando o DataLoader pede
        self.indices = []
        for s_idx, serie in enumerate(self.series):                       # para cada série (aeroporto)
            for i in range(len(serie) - self.janela - self.horizonte):    # desliza a janela pelo tempo
                self.indices.append((s_idx, i))                           # guarda: (qual aeroporto, qual posição)

    def __len__(self):
        return len(self.indices)              # total de exemplos de treino disponíveis

    def __getitem__(self, idx):
        s_idx, i = self.indices[idx]          # recupera qual aeroporto e qual posição no tempo
        serie = self.series[s_idx]            # pega a série normalizada do aeroporto

        # X: janela de 48h que o modelo usa para fazer a previsão
        X = torch.tensor(serie[i : i + self.janela], dtype=torch.float32)

        # y: as próximas 12h — apenas as 2 primeiras colunas (decolagens e passageiros) são o alvo
        y = torch.tensor(serie[i + self.janela : i + self.janela + self.horizonte, :2], dtype=torch.float32)

        return X, y                           # retorna o par (entrada, saída esperada)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA DE ATENÇÃO TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════

class AtencaoTemporal(nn.Module):
    """
    Mecanismo de atenção: aprende QUAIS horas do passado são mais importantes.
    Em vez de usar só o último estado do LSTM, pondera todos os estados pelo
    quanto cada hora contribui para a previsão.
    Ex: sexta às 18h pesa mais que terça às 3h para prever lotação.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # camada linear que atribui um score de importância a cada passo de tempo
        # *2 porque o LSTM é bidirecional — o estado oculto tem tamanho hidden*2
        self.atencao = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out tem shape: (batch, seq_len, hidden*2)
        scores = self.atencao(lstm_out)          # calcula score para cada hora: (batch, seq_len, 1)
        pesos  = torch.softmax(scores, dim=1)    # converte scores em probabilidades que somam 1
        ctx    = (lstm_out * pesos).sum(dim=1)   # média ponderada dos estados — resume a sequência num vetor
        return ctx                               # shape final: (batch, hidden*2)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO LSTM COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class LSTMAeroporto(nn.Module):
    """
    Arquitetura completa:
    [sequência 48h] → LSTM bidirecional → Atenção temporal → FC → [previsão 12h]

    Por que bidirecional? Lê a sequência de frente pra trás E de trás pra frente,
    capturando padrões como "queda antes do feriado" que só fazem sentido
    quando você vê o contexto em ambas as direções.
    """

    def __init__(self, n_features: int, horizonte: int):
        super().__init__()

        # LSTM bidirecional empilhado com 2 camadas
        self.lstm = nn.LSTM(
            input_size=n_features,                     # número de features de entrada (9 colunas)
            hidden_size=CONFIG["hidden_size"],          # tamanho do estado oculto (64)
            num_layers=CONFIG["num_layers"],            # número de LSTMs empilhados (2)
            batch_first=True,                          # espera tensores no formato (batch, seq, features)
            bidirectional=True,                        # processa a sequência nos dois sentidos
            dropout=CONFIG["dropout"] if CONFIG["num_layers"] > 1 else 0.0,  # dropout entre camadas (só com >1 layer)
        )

        self.atencao = AtencaoTemporal(CONFIG["hidden_size"])  # camada de atenção definida acima

        # cabeça de regressão: transforma o vetor de contexto na previsão final
        self.head = nn.Sequential(
            nn.Linear(CONFIG["hidden_size"] * 2, 128), # *2 porque o LSTM é bidirecional
            nn.ReLU(),                                  # ativação não-linear — introduz complexidade
            nn.Dropout(CONFIG["dropout"]),              # dropout para regularização durante o treino
            nn.Linear(128, horizonte * 2),              # 12h × 2 targets = 24 valores de saída
        )

        self.horizonte = horizonte  # guarda o horizonte para usar no reshape da saída

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)              # passa a sequência pelo LSTM; _ descarta os estados finais h_n e c_n
        ctx = self.atencao(lstm_out)            # aplica atenção para criar o vetor de contexto ponderado
        out = self.head(ctx)                    # projeta para a dimensão de saída (24 valores)
        return out.view(-1, self.horizonte, 2)  # reshape: (batch, 12h, 2 targets) — decolagens e passageiros


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÃO DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def treinar():
    log.info("Carregando features de aeroporto...")
    df = pl.read_parquet(FEATURES_DIR / "feat_aeroporto.parquet")  # lê o parquet gerado pelo script 02

    # remove registro com ano claramente errado encontrado nos dados brutos
    df = df.filter(pl.col("dt_partida_real").dt.year() <= 2025)

    # seleciona apenas os 20 aeroportos com mais voos — economiza RAM
    top_aeroportos = (
        df.group_by("sg_icao_origem")          # agrupa por código ICAO do aeroporto
          .agg(pl.len().alias("n"))            # conta quantos registros cada aeroporto tem
          .sort("n", descending=True)          # ordena do aeroporto com mais voos para o com menos
          .head(20)["sg_icao_origem"]          # pega os 20 primeiros
          .to_list()                           # converte para lista Python
    )
    df = df.filter(pl.col("sg_icao_origem").is_in(top_aeroportos))  # mantém só esses 20
    log.info(f"Aeroportos selecionados: {top_aeroportos}")

    # split temporal — NUNCA aleatório em séries temporais!
    # split aleatório vaza dados do futuro para o treino — modelo parece ótimo mas falha em produção
    df_treino = df.filter(pl.col("dt_partida_real").dt.year() <= 2023)  # treino: 2000–2023
    df_val    = df.filter(pl.col("dt_partida_real").dt.year() >= 2024)  # validação: 2024–2025

    gc.collect()  # força liberação de RAM antes de criar os datasets — evita crash por falta de memória

    # preenche nulos e NaN com 0 — sem isso a normalização gera NaN e quebra o treino
    df_treino = df_treino.with_columns([
        pl.col(c).fill_null(0).fill_nan(0) for c in FEATURES_SERIE  # aplica em cada coluna de feature
    ])
    df_val = df_val.with_columns([
        pl.col(c).fill_null(0).fill_nan(0) for c in FEATURES_SERIE
    ])

    # fit() APENAS nos dados de treino — se usar validação aqui o modelo "vê" o futuro
    scaler = StandardScaler()                                                # cria o normalizador
    scaler.fit(df_treino.select(FEATURES_SERIE).to_numpy().astype(np.float32))  # aprende média e desvio do treino

    ds_treino = AeroportoDataset(df_treino, scaler)  # cria dataset de treino com lazy loading
    ds_val    = AeroportoDataset(df_val,    scaler)  # cria dataset de validação (usa scaler do treino)

    dl_treino = DataLoader(
        ds_treino,
        batch_size=CONFIG["batch_size"],  # 32 exemplos por batch
        shuffle=True,                     # embaralha os batches a cada época — evita que o modelo aprenda a ordem
        num_workers=0,                    # 0 no Windows — valor > 0 causa crash de multiprocessing no Windows
        pin_memory=False,                 # True aceleraria transferência CPU→GPU mas pode causar erros no Windows
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=CONFIG["batch_size"] * 2,  # batch maior na validação (sem backward pass, usa menos memória)
        num_workers=0
    )

    device = CONFIG["device"]            # "cuda" (GPU) ou "cpu"
    log.info(f"Treinando em: {device} | {len(ds_treino):,} amostras de treino")

    # instancia o modelo e move todos os pesos para o device correto
    modelo = LSTMAeroporto(
        n_features=len(FEATURES_SERIE),  # 9 features de entrada
        horizonte=CONFIG["horizonte"],   # 12 horas a prever
    ).to(device)                         # .to(device) move pesos para GPU ou CPU

    # AdamW = Adam com weight decay — penaliza pesos grandes, melhor regularização que Adam puro
    otimizador = torch.optim.AdamW(
        modelo.parameters(),  # todos os parâmetros treináveis
        lr=CONFIG["lr"],      # taxa de aprendizado inicial (0.001)
        weight_decay=1e-4     # força os pesos a permanecerem pequenos (regularização L2)
    )

    # CosineAnnealingLR: decai o lr suavemente em forma de cosseno ao longo das épocas
    # evita que o modelo "pule" o mínimo do loss por ter lr muito alto no final
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        otimizador,
        T_max=CONFIG["epochs"],  # duração do ciclo de decaimento
        eta_min=1e-5             # lr mínimo que nunca será ultrapassado para baixo
    )

    # HuberLoss: mais robusto a outliers que MSE
    # para erros pequenos (|erro| < delta) usa MSE; para erros grandes usa MAE
    criterio = nn.HuberLoss(delta=1.0)

    melhor_val_loss = float("inf")  # inicializa com infinito — qualquer val loss vai ser melhor

    for epoch in range(1, CONFIG["epochs"] + 1):  # loop de 1 até 20

        # ── fase de treino ────────────────────────────────────────────────────
        modelo.train()       # ativa modo treino: dropout ligado, gradientes ativos
        perda_treino = 0.0   # acumulador da perda desta época

        for X, y in dl_treino:               # itera sobre cada batch
            X, y = X.to(device), y.to(device) # move tensores para GPU/CPU

            otimizador.zero_grad()             # OBRIGATÓRIO: zera gradientes do batch anterior
            pred = modelo(X)                   # forward pass — calcula a previsão
            loss = criterio(pred, y)           # calcula o erro entre previsão e valor real
            loss.backward()                    # backpropagation — calcula gradientes de todos os pesos

            # gradient clipping: corta gradientes que ficaram muito grandes
            # LSTMs são propensos a "explosão de gradientes" sem isso
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

            otimizador.step()                  # atualiza os pesos na direção de menor loss
            perda_treino += loss.item()        # .item() converte tensor escalar → float Python

        # ── fase de validação ─────────────────────────────────────────────────
        modelo.eval()      # ativa modo avaliação: dropout desligado
        perda_val = 0.0

        with torch.no_grad():           # sem cálculo de gradientes — economiza memória e tempo
            for X, y in dl_val:
                X, y = X.to(device), y.to(device)
                pred = modelo(X)
                perda_val += criterio(pred, y).item()

        perda_treino /= len(dl_treino)      # média por batch (não por exemplo)
        perda_val    /= max(len(dl_val), 1) # max(...,1) evita divisão por zero se val estiver vazio

        scheduler.step()  # atualiza a taxa de aprendizado conforme o cosine schedule

        log.info(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"Treino: {perda_treino:.4f} | Val: {perda_val:.4f}"
        )

        # salva o modelo apenas quando o val loss melhora — early stopping manual
        if perda_val < melhor_val_loss:
            melhor_val_loss = perda_val
            torch.save({
                "model_state":  modelo.state_dict(),    # pesos e biases do modelo
                "config":       CONFIG,                  # hiperparâmetros — necessários para recriar o modelo
                "scaler_mean":  scaler.mean_.tolist(),   # médias de cada feature — para normalizar novos dados
                "scaler_scale": scaler.scale_.tolist(),  # desvios de cada feature
                "features":     FEATURES_SERIE,          # nomes das features — garante a ordem correta
            }, MODELS_DIR / "modelo_aeroporto.pt")       # arquivo .pt = PyTorch checkpoint
            log.info(f"  ★ Melhor modelo salvo (val={perda_val:.4f})")

    log.info("✓ Treinamento concluído!")


# executa a função treinar() apenas quando o script é rodado diretamente
# se este arquivo for importado por outro script, treinar() NÃO é chamada automaticamente
if __name__ == "__main__":
    treinar()
