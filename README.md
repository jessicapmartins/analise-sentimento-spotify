# Análise de sentimento de avaliações do app Spotify

Classificação de avaliações do aplicativo Spotify como **positivas ou negativas** com Deep Learning (LSTM), mais um experimento de **IA generativa**: um Transformer que traduz as avaliações do inglês para o português.

Projeto em grupo da disciplina de Deep Learning da Pós-graduação em Analytics e Inteligência Artificial da FIA Business School (2024).

**Ferramentas:** Python · pandas · spaCy · NLTK · ydata-profiling · TensorFlow/Keras · Google Colab

---

## 1. Problema de negócio
Em apps de assinatura, a experiência do usuário pesa diretamente em churn e receita recorrente. As avaliações nas lojas de aplicativos são uma fonte contínua de feedback. Um modelo que as classifique automaticamente ajuda a priorizar melhorias de produto.

Durante a análise, o grupo percebeu que as avaliações tratam **do aplicativo, e não das músicas**. Por isso o escopo foi ajustado para refletir o que os dados realmente permitiam responder.

## 2. Dados
- **Fonte:** [Spotify App Reviews (Kaggle)](https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022/data). O arquivo não está neste repositório: baixe no Kaggle e salve como `data/reviews.csv`.
- **Volume:** 61.594 avaliações e 5 colunas (`Time_submitted`, `Review`, `Rating`, `Total_thumbsup`, `Reply`).
- **Colunas descartadas:**
  - `Time_submitted`: o problema não envolve tempo;
  - `Total_thumbsup`: cerca de 68% das avaliações sem curtidas;
  - `Reply`: quase 100% nula.

## 3. Preparação e análise exploratória
- **Variável resposta:** criada a partir do `Rating`. 4–5 estrelas viram positiva e 1–3 viram negativa. Com isso, as classes ficam equilibradas (51,4% negativas, 48,6% positivas).
- **Exploração:** perfil dos dados com ydata-profiling e nuvem de palavras.
- **Processamento de linguagem natural (spaCy):**
  - texto em minúsculas;
  - remoção de links, e-mails, menções e tags HTML;
  - conversão de emojis em texto;
  - tratamento de espaços;
  - remoção de stopwords (spaCy + NLTK) e lematização.
- **Tamanho das avaliações:** 75% têm até 21 palavras. Esse dado orientou o tamanho de entrada da rede.

## 4. Modelagem
Seis versões, ajustadas uma a uma:

| Versão | Principal mudança | Acurácia de validação (aprox.) |
|---|---|---|
| 0 – LSTM + Embedding | Sequências completas (mais de 200 palavras) | ~51% |
| 1 – LSTM + Embedding | Entrada limitada a 20 palavras | ~80%, com overfitting |
| 2 – LSTM + Embedding | 25 palavras, regularização L2, 50% da base | ~85% |
| 3 – LSTM + Embedding | Embedding maior, early stopping, checkpoint | ~85% |
| 4 – LSTM + Embedding | 100% da base | ~85% |
| 5 – Rede densa + Embedding | Arquitetura alternativa com L2 | ~83% |

**Modelo escolhido:**
- 50% da base e 25 palavras por avaliação;
- vocabulário de 13.630 palavras e embedding de dimensão 16;
- LSTM com 16 unidades e regularização L2;
- saída sigmoid.

**Resultado:** a acurácia subiu de aproximadamente **50% para 85%**. O maior ganho veio de limitar o número de palavras na entrada da rede.

## 5. IA generativa: tradução com Transformer
Segunda etapa: um Transformer encoder-decoder, implementado do zero em Keras, que traduz as avaliações do inglês para o português. A ideia é permitir a leitura do feedback em outro idioma.
- **Configuração final:** 100% dos dados, sequência de 40 tokens, vocabulário de 30 mil, embedding 512, 12 cabeças de atenção e dropout de 70%.
- **Resultado:** acurácia de validação perto de 77% por token. A validação piorou nas últimas épocas, o que indica overfitting.
- **Limitação:** o grupo concluiu que o ambiente do Colab limitava o tempo de treino. As traduções ainda não ficam fiéis em todos os casos.

## 6. Próximos passos
- Estudar o vocabulário para identificar as palavras que mais pesam na classificação.
- Testar outras arquiteturas e transfer learning, com modelos pré-treinados como BERT.
- Avaliar com precisão, recall e F1 além da acurácia, e analisar os erros por tipo de avaliação.

---

## Estrutura
```
├── notebooks/01_classificacao_sentimento_lstm.ipynb        # EDA, NLP e modelos LSTM
├── notebooks/02_traducao_transformer_ia_generativa.ipynb   # Transformer de tradução EN→PT
└── data/                                                   # coloque aqui o reviews.csv do Kaggle
```

**Autoria:** projeto em grupo, com participação de Jéssica Priscila de Oliveira Martins · [LinkedIn](https://www.linkedin.com/in/jessica-oliveira-martins/)
