# ML FastAPI — Scikit-learn Showcase

API REST profissional em **Python + FastAPI + scikit-learn** demonstrando
algoritmos clássicos e robustos de Machine Learning sobre o dataset Iris,
construída com **Clean Architecture** e padrão **MVC**.

---

## Highlights

- **FastAPI** com documentação OpenAPI automática (`/docs`, `/redoc`).
- **Clean Architecture** em camadas bem isoladas (controllers → services → ml/repositories).
- **MVC**: `models/` (schemas Pydantic), `views/` (presenters), `controllers/` (routers).
- **7 algoritmos sklearn** prontos:
  - Clássicos: Logistic Regression, KNN, Decision Tree, Naive Bayes
  - Robustos: Random Forest, Gradient Boosting, SVM (RBF + calibração para probabilidades)
- **Pipeline** com `StandardScaler` + estimador para cada modelo.
- **Avaliação completa**: acurácia, precisão, recall, F1 (macro), matriz de confusão, cross-validation 5-fold.
- **Visualizações** (matplotlib): scatter-matrix do dataset, comparação de algoritmos, matriz de confusão (heatmap), feature importance, curvas ROC OvR.
- **Persistência de modelos** via `joblib` (`artifacts/`).
- **Tratamento de erros** centralizado com exceções de domínio.
- **Testes** com `pytest` + `TestClient`.

---

## Arquitetura

```
app/
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
│   ├── health_controller.py
│   └── dependencies.py
├── services/                     # Application services (use-cases)
│   └── ml_service.py
├── ml/                           # Núcleo de ML (sem dependência de FastAPI)
│   ├── algorithms.py             # Registry de estimadores
│   ├── dataset.py                # Loader do dataset
│   ├── trainer.py                # Orquestração de treino
│   └── evaluator.py              # Métricas
└── repositories/                 # Persistência (joblib filesystem)
    └── model_repository.py
tests/
artifacts/                        # Modelos treinados (gitignored)
```

### Fluxo de uma requisição

```
HTTP → Controller → Service → ML domain (trainer/evaluator)
                              ↘ Repository (save/load)
                  ← View (presenter) ← Domain object
```

Cada seta atravessa uma fronteira clara. O domínio ML não conhece FastAPI;
controllers não conhecem sklearn; persistência é trocável.

---

## Como executar

### 1. Pré-requisitos
- Python **3.11+**

### 2. Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Subir a API

```bash
uvicorn app.main:app --reload
```

Acesse: <http://localhost:8000/docs>

---

## Endpoints

| Método | Rota                                          | Descrição                                       |
|--------|-----------------------------------------------|-------------------------------------------------|
| GET    | `/health`                                     | Liveness probe                                  |
| GET    | `/algorithms`                                 | Lista os 7 algoritmos disponíveis               |
| POST   | `/train`                                      | Treina um algoritmo e retorna métricas + id     |
| POST   | `/predict`                                    | Faz predições usando um modelo já treinado      |
| GET    | `/plots/dataset/scatter-matrix.png`           | Scatter matrix do Iris colorido por classe      |
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
    "per_class": { "setosa": {"precision":1.0,"recall":1.0,"f1":1.0,"support":10.0}, ... }
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

| Key                   | Categoria | Descrição                                              |
|-----------------------|-----------|--------------------------------------------------------|
| `logistic_regression` | classic   | Classificador linear probabilístico — baseline forte.  |
| `knn`                 | classic   | K-Nearest Neighbors (k=5).                             |
| `decision_tree`       | classic   | Árvore de decisão — modelo interpretável.              |
| `naive_bayes`         | classic   | Gaussian Naive Bayes.                                  |
| `random_forest`       | robust    | Ensemble de 200 árvores (bagging).                     |
| `gradient_boosting`   | robust    | Gradient Boosting sequencial.                          |
| `svm`                 | robust    | SVM com kernel RBF + `predict_proba`.                  |

Todos passam por uma `Pipeline` `StandardScaler → estimator`.

---

## Testes

```bash
pytest -v
```

Cobre: health, listagem, treino+predição de todos os 7 algoritmos, e
caminhos de erro (algoritmo inexistente, modelo não treinado, shape inválido).

---

## Decisões de design

- **Algorithm Registry**: adicionar um novo algoritmo é uma única entrada em
  `app/ml/algorithms.py`. O resto da aplicação não muda.
- **Service layer**: controllers ficam finos; orquestração e logging vivem
  no `MLService`.
- **Domain exceptions**: erros de domínio (`AlgorithmNotFoundError`,
  `ModelNotTrainedError`, `InvalidInputError`) são traduzidos para HTTP em
  `main.py`. Camadas inferiores não conhecem códigos HTTP.
- **Repository pattern**: hoje `joblib` em disco; amanhã S3/MLflow sem
  tocar em controllers/services.
- **Presenters**: a forma da resposta é decidida em um único lugar
  (`app/views/presenters.py`).

---

## Licença

MIT.
