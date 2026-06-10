"""Algorithm catalogue endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.controllers.dependencies import get_ml_service
from app.models.schemas import AlgorithmInfo
from app.services.ml_service import MLService
from app.views.presenters import present_algorithm

router = APIRouter(prefix="/algorithms", tags=["Algorithms"])


@router.get(
    "",
    response_model=list[AlgorithmInfo],
    summary="Listar algoritmos disponíveis",
    description=(
        "Retorna o catálogo completo de algoritmos suportados, com chave (`key`), "
        "nome legível, categoria e breve descrição.\n\n"
        "Use a `key` retornada aqui em **POST /train** e **POST /predict**.\n\n"
        "**Categorias:**\n"
        "- `classic`: algoritmos clássicos/didáticos "
        "(`logistic_regression`, `knn`, `decision_tree`, `naive_bayes`).\n"
        "- `robust`: algoritmos mais robustos/produção "
        "(`random_forest`, `gradient_boosting`, `svm`)."
    ),
    response_description="Lista com os 7 algoritmos disponíveis.",
)
def list_algorithms(
    service: MLService = Depends(get_ml_service),
) -> list[AlgorithmInfo]:
    """Return every algorithm available for training and prediction."""
    return [present_algorithm(spec) for spec in service.list_algorithms()]
