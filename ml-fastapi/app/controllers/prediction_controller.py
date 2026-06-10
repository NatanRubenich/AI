"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.controllers.dependencies import get_ml_service
from app.models.schemas import ErrorResponse, PredictRequest, PredictResponse
from app.services.ml_service import MLService
from app.views.presenters import present_predictions

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictResponse,
    summary="Prever a espécie de uma ou mais flores",
    description=(
        "Faz inferência usando um modelo **já treinado** via `POST /train`. "
        "Aceita predições em lote (várias amostras numa única chamada).\n\n"
        "**Ordem das features (sempre em centímetros):**\n"
        "1. `sepal_length` — comprimento da sépala\n"
        "2. `sepal_width` — largura da sépala\n"
        "3. `petal_length` — comprimento da pétala\n"
        "4. `petal_width` — largura da pétala\n\n"
        "**Resposta:** para cada instância, retorna a classe prevista "
        "(`setosa`, `versicolor` ou `virginica`), seu índice numérico e, quando "
        "o algoritmo suporta, as probabilidades por classe (soma ≈ 1).\n\n"
        "**Exemplos típicos:**\n"
        "- `[5.1, 3.5, 1.4, 0.2]` → quase certo `setosa`\n"
        "- `[6.7, 3.0, 5.2, 2.3]` → quase certo `virginica`\n"
        "- `[5.9, 3.0, 4.2, 1.5]` → tipicamente `versicolor`\n\n"
        "**Erros possíveis:**\n"
        "- `404`: algoritmo desconhecido.\n"
        "- `409`: modelo ainda não foi treinado — chame `POST /train` antes.\n"
        "- `422`: quantidade errada de features (precisa ser exatamente 4)."
    ),
    response_description="Predições com probabilidades por classe.",
    responses={
        404: {"model": ErrorResponse, "description": "Algoritmo desconhecido."},
        409: {"model": ErrorResponse, "description": "Modelo não foi treinado ainda."},
        422: {"model": ErrorResponse, "description": "Shape de input inválido."},
    },
)
def predict(
    payload: PredictRequest,
    service: MLService = Depends(get_ml_service),
) -> PredictResponse:
    """Run inference for one or more instances against a trained model."""
    raw = service.predict(payload.algorithm, payload.instances)
    return present_predictions(payload.algorithm, raw)
