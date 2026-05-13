# 📊 Previsão de Churn com Framework SEMMA

Modelo preditivo de Machine Learning para identificar usuários com alta probabilidade de cancelamento (Churn), desenvolvido com o framework analítico **SEMMA**.

---

## 🗂 Base de Dados

**[ABT Churn Journey](https://www.kaggle.com/datasets/teocalvo/analytical-base-table-churn/data)** — Analytical Base Table com histórico de transações e comportamento de usuários.

| Coluna | Descrição |
|--------|-----------|
| `dtRef` | Data de referência da safra |
| `idUsuario` | Identificador único do usuário |
| `flagChurn` | Variável target (1 = Churn, 0 = Não Churn) |
| `qtdeTransacoes` | Quantidade de transações |
| `qtdeDias` | Quantidade de dias ativos |
| `mediaTransacoesDias` | Média de transações por dia |
| `saldoPontos` | Saldo de pontos acumulados |
| `qtdePontosPos` | Quantidade de pontos positivos |
| ... | Demais variáveis comportamentais |

---

## 🔁 Framework SEMMA

### 1. S — Sample (Amostragem)

- **Base OOT (Out-of-Time):** Último mês disponível (`2025-04-01`) separado como base de validação temporal. Não foi utilizado durante o treinamento.
- **Train/Test Split:** 80% treino / 20% teste com **estratificação** pela variável `flagChurn`.

| Base | Proporção de Churn |
|------|--------------------|
| Treino | 46,89% |
| Teste | 46,87% |
| OOT | 50,50% |

---

### 2. E — Explore (Exploração)

> ⚠️ Toda exploração foi realizada **apenas na base de treino**, evitando data leakage.

- **Qualidade dos dados:** Nenhuma variável apresentou valores nulos.
- **Análise Bivariada:** Usuários com Churn apresentaram, em média, menos transações, menos dias ativos e menor saldo de pontos.
- **Feature Importance (Decision Tree):** Identificadas as variáveis com maior poder preditivo para ranquear as mais promissoras.

---

### 3. M — Modify (Modificação)

Transformações encapsuladas em um **sklearn Pipeline** para garantir consistência entre treino, teste e OOT:

- **Variáveis numéricas:** Imputação pela mediana + Discretização em 5 bins (KBinsDiscretizer, estratégia quantil)
- **Variáveis categóricas:** Imputação com constante `'missing'` + One-Hot Encoding

---

### 4. M — Model (Modelagem)

**Algoritmo:** Random Forest Classifier

**Otimização de hiperparâmetros** via `RandomizedSearchCV` (n_iter=20, cv=5, scoring=ROC AUC):

| Hiperparâmetro | Melhor Valor |
|----------------|--------------|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `min_samples_split` | 10 |
| `class_weight` | None |

---

### 5. A — Assess (Avaliação)

#### Performance — ROC AUC

| Base | AUC |
|------|-----|
| Treino | **0.8334** |
| Teste | **0.8281** |
| OOT | **0.8441** |

> ✅ O modelo não apresenta sinais de overfitting: os resultados são estáveis entre treino, teste e OOT.

#### Análise de Ganhos (Lift — Base Teste)

| Decil (% usuários abordados) | % Churners capturados |
|------------------------------|-----------------------|
| Top 10% | 18,3% |
| Top 20% | 36,3% |
| Top 30% | 49,9% |

> O modelo captura **quase 50% de todos os churners** abordando apenas 30% dos usuários com maior risco, demonstrando ganho expressivo em relação a uma seleção aleatória.

---

## 🛠 Stack Tecnológica

- **Python 3.12**
- `pandas`, `numpy`
- `scikit-learn` (Pipeline, RandomForestClassifier, RandomizedSearchCV, KBinsDiscretizer, OneHotEncoder)
- `matplotlib`, `seaborn`
- `pickle`

---

## 🚀 Como Executar

1. Faça o download do dataset no [Kaggle](https://www.kaggle.com/datasets/teocalvo/analytical-base-table-churn/data) e salve como `abt_churn.csv` na raiz do projeto.
2. Abra o notebook `modelSEMMA.ipynb` no Google Colab ou Jupyter.
3. Execute as células em ordem.
4. O modelo final será salvo como `modelo_churn_final.pkl`.

---

## 📁 Estrutura do Projeto

```
.
├── abt_churn.csv              # Dataset (baixar do Kaggle)
├── modelSEMMA.ipynb           # Notebook principal com o pipeline SEMMA
├── modelo_churn_final.pkl     # Modelo serializado (gerado após execução)
└── README.md
```

---

## 📌 Referências

- [Kaggle — ABT Churn Journey](https://www.kaggle.com/datasets/teocalvo/analytical-base-table-churn/data)
- [Documentação scikit-learn](https://scikit-learn.org/stable/)
