"""Training endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.controllers.dependencies import get_ml_service
from app.models.schemas import ErrorResponse, TrainRequest, TrainResponse
from app.services.ml_service import MLService
from app.views.presenters import present_training

router = APIRouter(prefix="/train", tags=["Training"])


@router.post(
    "",
    response_model=TrainResponse,
    summary="Treinar um algoritmo no dataset Iris",
    description=(
        "Treina o algoritmo informado sobre o dataset Iris e devolve métricas de avaliação "
        "completas + um `model_id` para usar em predições.\n\n"
        "**O que acontece internamente:**\n"
        "1. Carrega o dataset Iris (150 amostras, 4 features, 3 classes).\n"
        "2. Faz `train_test_split` estratificado (mantém proporção das classes) usando "
        "`test_size` e `random_state`.\n"
        "3. Monta uma `Pipeline` (StandardScaler + estimador) e a ajusta no conjunto de treino.\n"
        "4. Avalia no conjunto de teste (acurácia, precision/recall/F1 macro, matriz de confusão, "
        "métricas por classe) **e** executa validação cruzada 5-fold no dataset completo "
        "(`cv_mean_accuracy` ± `cv_std_accuracy`) para estimar a estabilidade do modelo.\n"
        "5. Persiste o modelo treinado em `artifacts/{algorithm}.joblib`.\n\n"
        "**Parâmetros aceitáveis:**\n"
        "- `algorithm`: chave de **GET /algorithms**.\n"
        "- `test_size`: `0.2` é o padrão recomendado. Faixa: `(0.05, 0.5)`.\n"
        "- `random_state`: `42` é o padrão. Qualquer inteiro ≥ 0 fixa a aleatoriedade.\n\n"
        "**Erros possíveis:**\n"
        "- `404`: algoritmo desconhecido.\n"
        "- `422`: parâmetros inválidos (test_size fora da faixa, etc.)."
    ),
    response_description="Algoritmo treinado com métricas completas e `model_id`.",
    responses={
        404: {"model": ErrorResponse, "description": "Algoritmo não existe no registry."},
        422: {"model": ErrorResponse, "description": "Parâmetros inválidos."},
    },
)
def train_algorithm(
    payload: TrainRequest,
    service: MLService = Depends(get_ml_service),
) -> TrainResponse:
    """Fit `payload.algorithm` and return evaluation metrics + model id."""
    result, model_id = service.train_algorithm(
        algorithm_key=payload.algorithm,
        test_size=payload.test_size,
        random_state=payload.random_state,
    )
    return present_training(result, model_id)
