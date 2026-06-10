"""
Pydantic schemas (API contracts).

Schemas are the boundary types between HTTP and the application core.
They MUST NOT depend on sklearn or any persistence detail.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# --- Algorithms --------------------------------------------------------------


class AlgorithmInfo(BaseModel):
    """Descritor de um algoritmo disponível na API."""

    key: str = Field(..., description="Identificador usado em /train e /predict.", examples=["random_forest"])
    display_name: str = Field(..., description="Nome legível do algoritmo.", examples=["Random Forest"])
    category: Literal["classic", "robust"] = Field(..., description="`classic` = algoritmos didáticos; `robust` = algoritmos de produção.")
    description: str = Field(..., description="Explicação curta do funcionamento do algoritmo.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "key": "random_forest",
                    "display_name": "Random Forest",
                    "category": "robust",
                    "description": "Ensemble of decorrelated decision trees (bagging).",
                }
            ]
        }
    }


# --- Training ----------------------------------------------------------------


class TrainRequest(BaseModel):
    """Parâmetros para treinar um algoritmo."""

    algorithm: str = Field(
        ...,
        description=(
            "Chave do algoritmo a treinar. Valores válidos: "
            "`logistic_regression`, `knn`, `decision_tree`, `naive_bayes`, "
            "`random_forest`, `gradient_boosting`, `svm`. "
            "Consulte `GET /algorithms` para a lista completa com descrições."
        ),
        examples=["random_forest"],
    )
    test_size: float = Field(
        default=0.2,
        gt=0.05,
        lt=0.5,
        description=(
            "Fração do dataset reservada para teste (não vista no treino). "
            "Padrão: 0.2 (20%). Faixa aceitável: 0.05 < test_size < 0.5."
        ),
        examples=[0.2],
    )
    random_state: int = Field(
        default=42,
        ge=0,
        description=(
            "Semente da aleatoriedade para reprodutibilidade do split. "
            "Qualquer inteiro ≥ 0 serve; mudar o valor produz resultados diferentes."
        ),
        examples=[42],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "algorithm": "random_forest",
                    "test_size": 0.2,
                    "random_state": 42,
                }
            ]
        }
    }


class PerClassMetrics(BaseModel):
    """Métricas por classe no conjunto de teste."""

    precision: float = Field(..., description="TP / (TP + FP) — das que o modelo disse ser desta classe, qual fração era mesmo?")
    recall: float = Field(..., description="TP / (TP + FN) — das que realmente eram desta classe, qual fração o modelo pegou?")
    f1: float = Field(..., description="Média harmônica entre precision e recall.")
    support: float = Field(..., description="Número de amostras desta classe no conjunto de teste.")


class EvaluationSchema(BaseModel):
    """Relatório agregado de avaliação do modelo."""

    accuracy: float = Field(..., description="Acurácia no conjunto de teste (acertos / total).")
    precision_macro: float = Field(..., description="Precision média entre as classes (sem ponderação por suporte).")
    recall_macro: float = Field(..., description="Recall médio entre as classes.")
    f1_macro: float = Field(..., description="F1 médio entre as classes.")
    cv_mean_accuracy: float = Field(..., description="Média da acurácia em validação cruzada 5-fold no dataset completo.")
    cv_std_accuracy: float = Field(..., description="Desvio-padrão da acurácia entre os 5 folds (estabilidade do modelo).")
    confusion_matrix: list[list[int]] = Field(
        ...,
        description="Matriz de confusão: linhas = classe real, colunas = classe prevista. A diagonal são acertos.",
    )
    per_class: dict[str, PerClassMetrics] = Field(..., description="Métricas detalhadas por classe.")


class TrainResponse(BaseModel):
    """Resultado de um treino: metadados do algoritmo + métricas + id do modelo persistido."""

    algorithm: AlgorithmInfo
    n_train: int = Field(..., description="Número de amostras usadas no treino.", examples=[120])
    n_test: int = Field(..., description="Número de amostras reservadas para teste.", examples=[30])
    feature_names: list[str] = Field(..., description="Ordem esperada das features em /predict.")
    target_names: list[str] = Field(..., description="Nomes das classes na ordem dos índices.")
    evaluation: EvaluationSchema
    model_id: str = Field(
        ...,
        description="Identificador do modelo persistido. Use este valor como `algorithm` em /predict.",
        examples=["random_forest"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "algorithm": {
                        "key": "random_forest",
                        "display_name": "Random Forest",
                        "category": "robust",
                        "description": "Ensemble of decorrelated decision trees (bagging).",
                    },
                    "n_train": 120,
                    "n_test": 30,
                    "feature_names": [
                        "sepal length (cm)",
                        "sepal width (cm)",
                        "petal length (cm)",
                        "petal width (cm)",
                    ],
                    "target_names": ["setosa", "versicolor", "virginica"],
                    "evaluation": {
                        "accuracy": 0.9667,
                        "precision_macro": 0.9697,
                        "recall_macro": 0.9667,
                        "f1_macro": 0.9665,
                        "cv_mean_accuracy": 0.9667,
                        "cv_std_accuracy": 0.0211,
                        "confusion_matrix": [[10, 0, 0], [0, 10, 0], [0, 1, 9]],
                        "per_class": {
                            "setosa": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 10},
                            "versicolor": {"precision": 0.91, "recall": 1.0, "f1": 0.95, "support": 10},
                            "virginica": {"precision": 1.0, "recall": 0.9, "f1": 0.95, "support": 10},
                        },
                    },
                    "model_id": "random_forest",
                }
            ]
        }
    }


# --- Prediction --------------------------------------------------------------


class PredictRequest(BaseModel):
    """Requisição de predição (individual ou em lote)."""

    algorithm: str = Field(
        ...,
        description=(
            "Chave do algoritmo já treinado (mesmo valor de `model_id` retornado por /train). "
            "O modelo precisa ter sido treinado previamente, caso contrário retorna 409."
        ),
        examples=["random_forest"],
    )
    instances: list[list[float]] = Field(
        ...,
        description=(
            "Lista de vetores de features. Cada vetor precisa ter exatamente 4 valores na ordem: "
            "`[sepal_length, sepal_width, petal_length, petal_width]` (em centímetros)."
        ),
        min_length=1,
        examples=[[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]],
    )

    @field_validator("instances")
    @classmethod
    def _no_empty_vectors(cls, v: list[list[float]]) -> list[list[float]]:
        if any(len(row) == 0 for row in v):
            raise ValueError("instances cannot contain empty feature vectors")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "algorithm": "random_forest",
                    "instances": [
                        [5.1, 3.5, 1.4, 0.2],
                        [6.7, 3.0, 5.2, 2.3],
                    ],
                }
            ]
        }
    }


class Prediction(BaseModel):
    """Predição para uma única instância."""

    predicted_class: str = Field(..., description="Nome da classe prevista.", examples=["setosa"])
    predicted_index: int = Field(..., description="Índice numérico da classe prevista (0, 1 ou 2).", examples=[0])
    probabilities: dict[str, float] | None = Field(
        default=None,
        description=(
            "Probabilidade estimada para cada classe (soma ≈ 1). "
            "Pode ser `null` se o algoritmo não suportar `predict_proba`."
        ),
    )


class PredictResponse(BaseModel):
    """Resposta de predição: uma entrada para cada instância recebida."""

    algorithm: str = Field(..., description="Chave do algoritmo usado para esta predição.")
    predictions: list[Prediction]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "algorithm": "random_forest",
                    "predictions": [
                        {
                            "predicted_class": "setosa",
                            "predicted_index": 0,
                            "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
                        },
                        {
                            "predicted_class": "virginica",
                            "predicted_index": 2,
                            "probabilities": {"setosa": 0.0, "versicolor": 0.02, "virginica": 0.98},
                        },
                    ],
                }
            ]
        }
    }


# --- Misc --------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Resposta da verificação de saúde (liveness probe)."""

    status: Literal["ok"]
    app: str = Field(..., examples=["ML FastAPI - Scikit-learn Showcase"])
    version: str = Field(..., examples=["1.0.0"])


class ErrorResponse(BaseModel):
    """Formato padrão de respostas de erro."""

    detail: str = Field(..., description="Mensagem explicando o erro.")
