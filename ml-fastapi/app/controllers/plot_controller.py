"""Visualization endpoints (PNG)."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, Response

from app.services.plot_service import PlotService

router = APIRouter(prefix="/plots", tags=["Plots"])


@lru_cache(maxsize=1)
def get_plot_service() -> PlotService:
    return PlotService()


def _png(data: bytes) -> Response:
    return Response(content=data, media_type="image/png")


_PNG_OK = {"content": {"image/png": {}}}


@router.get(
    "/dataset/scatter-matrix.png",
    summary="Matriz de dispersão do dataset Iris",
    description=(
        "Gera uma **matriz de dispersão (scatter matrix)** do dataset Iris, com cada "
        "ponto colorido pela espécie. Cada célula compara duas features; a diagonal "
        "mostra o histograma de cada feature.\n\n"
        "**Como ler:** procure pares de features onde as cores formam grupos bem "
        "separados — esses são pares onde as classes são fáceis de distinguir. "
        "Para o Iris, `petal_length × petal_width` separa muito bem.\n\n"
        "**Não requer treino prévio** — usa apenas o dataset bruto."
    ),
    response_description="Imagem PNG da matriz de dispersão.",
    responses={200: _PNG_OK},
)
def scatter_matrix(service: PlotService = Depends(get_plot_service)) -> Response:
    return _png(service.scatter_matrix_png())


@router.get(
    "/comparison.png",
    summary="Comparar todos os algoritmos lado a lado",
    description=(
        "Treina os **7 algoritmos** com parâmetros padrão e gera um gráfico de barras "
        "comparando três métricas para cada um:\n"
        "- **Acurácia** no conjunto de teste\n"
        "- **F1 (macro)** no conjunto de teste\n"
        "- **Acurácia média** em validação cruzada 5-fold\n\n"
        "Útil para ver de relance qual algoritmo se sai melhor no Iris.\n\n"
        "**Observação:** este endpoint executa 7 treinos — pode levar alguns segundos."
    ),
    response_description="Imagem PNG do bar chart comparativo.",
    responses={200: _PNG_OK},
)
def comparison(service: PlotService = Depends(get_plot_service)) -> Response:
    return _png(service.comparison_png())


@router.get(
    "/{algorithm}/confusion-matrix.png",
    summary="Matriz de confusão de um modelo treinado",
    description=(
        "Heatmap da matriz de confusão no conjunto de teste do algoritmo informado.\n\n"
        "**Como ler:** linhas = classe real, colunas = classe prevista. A **diagonal** "
        "mostra os acertos; valores fora da diagonal mostram **quais classes o modelo "
        "está confundindo** entre si.\n\n"
        "**Requer treino prévio** via `POST /train`."
    ),
    response_description="Imagem PNG do heatmap.",
    responses={
        200: _PNG_OK,
        404: {"description": "Algoritmo desconhecido."},
        409: {"description": "Modelo não foi treinado ainda."},
    },
)
def confusion_matrix(
    algorithm: str, service: PlotService = Depends(get_plot_service)
) -> Response:
    return _png(service.confusion_matrix_png(algorithm))


@router.get(
    "/{algorithm}/feature-importance.png",
    summary="Importância das features (apenas tree-based)",
    description=(
        "Bar chart horizontal ordenado mostrando quanto cada feature contribuiu "
        "para as decisões do modelo (atributo `feature_importances_` do sklearn).\n\n"
        "**Apenas para modelos baseados em árvore:** `decision_tree`, `random_forest`, "
        "`gradient_boosting`. Para outros algoritmos retorna **422**.\n\n"
        "**Requer treino prévio** via `POST /train`."
    ),
    response_description="Imagem PNG do bar chart de importâncias.",
    responses={
        200: _PNG_OK,
        404: {"description": "Algoritmo desconhecido."},
        409: {"description": "Modelo não foi treinado ainda."},
        422: {"description": "Algoritmo não suporta feature_importances_."},
    },
)
def feature_importance(
    algorithm: str, service: PlotService = Depends(get_plot_service)
) -> Response:
    return _png(service.feature_importance_png(algorithm))


@router.get(
    "/{algorithm}/roc-curve.png",
    summary="Curvas ROC (Um-vs-Resto) do modelo treinado",
    description=(
        "Gera curvas **ROC One-vs-Rest** para problema multiclasse: uma curva por "
        "classe (classe `i` vs todas as outras), com **AUC** anotada na legenda.\n\n"
        "**Como ler:** quanto mais a curva se aproxima do canto superior esquerdo, "
        "melhor o modelo separa aquela classe das demais. **AUC = 1.0** é perfeito, "
        "**AUC = 0.5** é sorteio.\n\n"
        "**Requer treino prévio** + algoritmo com suporte a `predict_proba`. "
        "Todos os 7 algoritmos do projeto suportam."
    ),
    response_description="Imagem PNG das curvas ROC.",
    responses={
        200: _PNG_OK,
        404: {"description": "Algoritmo desconhecido."},
        409: {"description": "Modelo não foi treinado ainda."},
    },
)
def roc_curve(
    algorithm: str, service: PlotService = Depends(get_plot_service)
) -> Response:
    return _png(service.roc_curve_png(algorithm))
