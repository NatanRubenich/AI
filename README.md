# ML FastAPI — Classificador de Flores Iris

API REST em **Python + FastAPI + scikit-learn** que aprende a **classificar
espécies de flores** do gênero *Iris* a partir de 4 medidas físicas. O projeto
treina, compara e serve **7 algoritmos clássicos de Machine Learning**,
expondo tudo via HTTP e visualizações em PNG — organizado em
**Clean Architecture + MVC**.

<p align="center">
  <img src="https://content.codecademy.com/programs/machine-learning/k-means/iris.svg"
       alt="As três espécies de Iris: setosa, versicolor e virginica"
       width="640"/>
</p>

---

## Qual o propósito do algoritmo?

**Problema:** dadas 4 medidas de uma flor (comprimento e largura de sépala e
pétala, em centímetros), identificar a qual das **3 espécies** ela pertence:

- **Iris setosa** — pétalas pequenas e curtas, sépalas largas
- **Iris versicolor** — medidas intermediárias
- **Iris virginica** — pétalas grandes e alongadas

É um problema de **classificação multiclasse supervisionada**: o modelo aprende
a partir de exemplos rotulados (150 flores já identificadas por botânicos) e
generaliza para flores novas que nunca viu.

**Por que esse problema importa, didaticamente:**

- É o **"Hello World" do Machine Learning** — usado desde 1936 (Ronald Fisher) e
  consagrado pela comunidade.
- Demonstra **todos os blocos fundamentais de um pipeline de ML real** —
  carregar dados, separar treino/teste, normalizar features, treinar diferentes
  famílias de algoritmos, avaliar com múltiplas métricas e visualizar resultados.
- Os modelos atingem **~95–97% de acurácia**, com erros restritos à fronteira
  entre `versicolor` e `virginica` — perfeito para discutir matriz de confusão,
  ROC e *trade-offs*.

**O que a API permite:**

1. **Treinar** qualquer um dos 7 algoritmos e ver suas métricas
2. **Comparar** algoritmos lado a lado
3. **Prever** a espécie de novas flores (em lote, com probabilidades)
4. **Visualizar** o dataset e o comportamento dos modelos (matriz de confusão,
   importância de features, curvas ROC)

---

## Highlights técnicos

- **FastAPI** com documentação OpenAPI rica (Swagger em `/docs`, ReDoc em `/redoc`).
- **Clean Architecture** em camadas isoladas (controllers → services → ml/repositories).
- **MVC**: `models/` (schemas Pydantic), `views/` (presenters), `controllers/` (routers).
- **7 algoritmos sklearn** prontos:
  - **Clássicos:** Logistic Regression, KNN, Decision Tree, Naive Bayes
  - **Robustos:** Random Forest, Gradient Boosting, SVM (RBF + calibração de probabilidades)
- **Pipeline** com `StandardScaler` + estimador para cada modelo.
- **Avaliação completa**: acurácia, precision, recall, F1 (macro), matriz de confusão, validação cruzada 5-fold.
- **Visualizações** (matplotlib): scatter-matrix do dataset, comparativo de algoritmos, matriz de confusão (heatmap), feature importance, curvas ROC OvR.
- **Persistência de modelos** via `joblib` em `artifacts/`.
- **Tratamento de erros** centralizado com exceções de domínio (404/409/422).
- **Testes** com `pytest` + `TestClient` — **26 testes** passando.

---

## Dataset Iris

Os dados vêm **embutidos no scikit-learn** (`sklearn.datasets.load_iris`):
zero setup, sem download, sem CSV.

| Característica | Valor |
|----------------|-------|
| Amostras       | **150** (50 por espécie — perfeitamente balanceado) |
| Features       | **4** numéricas em cm |
| Classes        | **3** espécies (`setosa`, `versicolor`, `virginica`) |
| Origem         | Ronald A. Fisher, 1936 (medidas de Edgar Anderson) |
| Característica | Sem valores faltantes, sem outliers significativos |

**Features (na ordem esperada por `/predict`):**

1. `sepal_length` — comprimento da sépala
2. `sepal_width` — largura da sépala
3. `petal_length` — comprimento da pétala
4. `petal_width` — largura da pétala

> **Quer trocar o dataset?** O carregamento está isolado em
> `ml-fastapi/app/ml/dataset.py`. Mudar essa única função (para ler CSV,
> banco, S3, etc.) não afeta nenhuma outra camada.

---

## Arquitetura

```
ml-fastapi/
└── app/
    ├── main.py                       # Application factory (FastAPI)
    ├── core/                         # Config, logging, exceptions
    │   ├── config.py
    │   ├── logging.py
    │   └── exceptions.py
    ├── models/                       # M (Pydantic schemas — contratos da API)
    │   └── schemas.py
    ├── views/                        # V (presenters: domain → schema)
    │   └── presenters.py
    ├── controllers/                  # C (routers FastAPI)
    │   ├── algorithm_controller.py
    │   ├── training_controller.py
    │   ├── prediction_controller.py
    │   ├── plot_controller.py
    │   ├── health_controller.py
    │   └── dependencies.py
    ├── services/                     # Application services (use-cases)
    │   ├── ml_service.py
    │   └── plot_service.py
    ├── ml/                           # Núcleo de ML (sem dependência de FastAPI)
    │   ├── algorithms.py             # Registry de estimadores
    │   ├── dataset.py                # Loader do dataset
    │   ├── trainer.py                # Orquestração de treino
    │   ├── evaluator.py              # Métricas
    │   └── plots.py                  # Plot primitives (matplotlib → PNG bytes)
    └── repositories/                 # Persistência (joblib filesystem)
        └── model_repository.py
tests/
artifacts/                            # Modelos treinados (gitignored)
```

### Fluxo de uma requisição

```
HTTP → Controller → Service → ML domain (trainer/evaluator/plots)
                              ↘ Repository (save/load)
                  ← View (presenter) ← Domain object
```

Cada seta atravessa uma fronteira clara. O domínio ML não conhece FastAPI;
controllers não conhecem sklearn; persistência é trocável.

---

## Como executar

### 1. Pré-requisitos

- Python **3.11+** (testado em 3.14)

### 2. Instalação

```bash
cd ml-fastapi
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Subir a API

```bash
uvicorn app.main:app --reload
```

Abra **<http://localhost:8000/docs>** para o Swagger interativo.

---

## Endpoints

| Método | Rota                                          | Descrição                                       |
|--------|-----------------------------------------------|-------------------------------------------------|
| GET    | `/health`                                     | Liveness probe                                  |
| GET    | `/algorithms`                                 | Lista os 7 algoritmos disponíveis               |
| POST   | `/train`                                      | Treina um algoritmo e retorna métricas + id     |
| POST   | `/predict`                                    | Faz predições usando um modelo já treinado      |
| GET    | `/plots/dataset/scatter-matrix.png`           | Matriz de dispersão do Iris colorida por classe |
| GET    | `/plots/comparison.png`                       | Gráfico comparando os 7 algoritmos              |
| GET    | `/plots/{algorithm}/confusion-matrix.png`     | Matriz de confusão (heatmap) do modelo treinado |
| GET    | `/plots/{algorithm}/feature-importance.png`   | Importância de features (modelos tree-based)    |
| GET    | `/plots/{algorithm}/roc-curve.png`            | Curvas ROC one-vs-rest                          |

### Exemplo: treinar Random Forest

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "random_forest", "test_size": 0.2, "random_state": 42}'
```

Resposta (resumida):

```json
{
  "algorithm": {
    "key": "random_forest",
    "display_name": "Random Forest",
    "category": "robust",
    "description": "Ensemble of decorrelated decision trees (bagging)."
  },
  "n_train": 120,
  "n_test": 30,
  "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
  "target_names": ["setosa", "versicolor", "virginica"],
  "evaluation": {
    "accuracy": 0.9667,
    "f1_macro": 0.9665,
    "cv_mean_accuracy": 0.9667,
    "cv_std_accuracy": 0.0211,
    "confusion_matrix": [[10,0,0],[0,10,0],[0,1,9]],
    "per_class": { "setosa": {"precision":1.0,"recall":1.0,"f1":1.0,"support":10.0} }
  },
  "model_id": "random_forest"
}
```

### Exemplo: predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"random_forest","instances":[[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3]]}'
```

```json
{
  "algorithm": "random_forest",
  "predictions": [
    {"predicted_class":"setosa","predicted_index":0,"probabilities":{"setosa":0.99,"versicolor":0.01,"virginica":0.0}},
    {"predicted_class":"virginica","predicted_index":2,"probabilities":{"setosa":0.0,"versicolor":0.02,"virginica":0.98}}
  ]
}
```

---

## Algoritmos disponíveis

| Key                   | Categoria | Intuição                                                              |
|-----------------------|-----------|-----------------------------------------------------------------------|
| `logistic_regression` | classic   | Classificador linear probabilístico — baseline forte.                 |
| `knn`                 | classic   | K-Nearest Neighbors (k=5) — vota nos 5 vizinhos mais próximos.        |
| `decision_tree`       | classic   | Árvore de perguntas "feature > threshold?" — totalmente interpretável.|
| `naive_bayes`         | classic   | Gaussian Naive Bayes — probabilístico, assume features independentes. |
| `random_forest`       | robust    | Ensemble de 200 árvores (bagging) — robusto e dá feature importance.  |
| `gradient_boosting`   | robust    | Árvores sequenciais corrigindo erros das anteriores (boosting).       |
| `svm`                 | robust    | SVM kernel RBF + `CalibratedClassifierCV` para probabilidades.        |

Todos passam por uma `Pipeline` `StandardScaler → estimator`.

---

## Métricas calculadas (em cada `/train`)

- **Acurácia** no conjunto de teste
- **Precision / Recall / F1** por classe + médias macro
- **Matriz de confusão** (linhas = real, colunas = previsto)
- **Validação cruzada 5-fold** (média ± desvio) — mede estabilidade do modelo

---

## Testes

```bash
cd ml-fastapi
pytest -v
```

**26 testes** cobrindo: health check, catálogo de algoritmos, treino +
predição dos 7 algoritmos, todos os 5 endpoints de plot, e caminhos de erro
(404 algoritmo inexistente, 409 modelo não treinado, 422 shape inválido).

---

## Decisões de design

- **Algorithm Registry**: adicionar um novo algoritmo é uma única entrada em
  `app/ml/algorithms.py`. Resto da aplicação não muda.
- **Service layer**: controllers ficam finos; orquestração e logging vivem
  nos services.
- **Domain exceptions**: erros de domínio (`AlgorithmNotFoundError`,
  `ModelNotTrainedError`, `InvalidInputError`) são traduzidos para HTTP em
  `main.py`. Camadas inferiores não conhecem códigos HTTP.
- **Repository pattern**: hoje `joblib` em disco; amanhã S3/MLflow sem
  tocar em controllers/services.
- **Presenters**: a forma da resposta JSON é decidida em um único lugar
  (`app/views/presenters.py`).
- **Plot primitives puros**: `app/ml/plots.py` recebe arrays e devolve PNG
  bytes — não conhece FastAPI nem persistência. Novo gráfico = 1 função.

---

## Créditos da imagem

Ilustração das três espécies de *Iris* por **Codecademy**
([fonte](https://content.codecademy.com/programs/machine-learning/k-means/iris.svg)).

---

## Licença

MIT.
