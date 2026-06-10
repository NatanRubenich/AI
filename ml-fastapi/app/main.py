"""
FastAPI application entrypoint.

Wires controllers (routers), exception handlers, and middleware.
Following clean-architecture, this module is the only place that knows
about every other layer.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.controllers import (
    algorithm_controller,
    health_controller,
    plot_controller,
    prediction_controller,
    training_controller,
)
from app.core.config import settings
from app.core.exceptions import (
    AlgorithmNotFoundError,
    InvalidInputError,
    ModelNotTrainedError,
)
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Verificação de saúde do servidor (liveness probe).",
    },
    {
        "name": "Algorithms",
        "description": "Catálogo dos 7 algoritmos disponíveis para treino e predição.",
    },
    {
        "name": "Training",
        "description": (
            "Treina um algoritmo no dataset Iris e devolve métricas completas + "
            "`model_id` para uso em predições. O modelo treinado é persistido em disco."
        ),
    },
    {
        "name": "Prediction",
        "description": (
            "Faz inferência (individual ou em lote) usando um modelo já treinado. "
            "Retorna a classe prevista e probabilidades por classe."
        ),
    },
    {
        "name": "Plots",
        "description": (
            "Visualizações geradas como imagens PNG: matriz de dispersão do dataset, "
            "comparativo de algoritmos, matriz de confusão, importância das features "
            "e curvas ROC."
        ),
    },
]


LONG_DESCRIPTION = """
API REST para demonstrar **algoritmos de Machine Learning do scikit-learn** sobre o
dataset clássico **Iris**, organizada em **Clean Architecture + MVC**.

## Fluxo típico de uso

1. **`GET /algorithms`** — descubra quais algoritmos estão disponíveis.
2. **`POST /train`** — treine um algoritmo e receba métricas + `model_id`.
3. **`POST /predict`** — use o `model_id` para prever a espécie de novas flores.
4. **`GET /plots/...`** — explore visualmente o dataset e o modelo treinado.

## Algoritmos

- **Clássicos:** `logistic_regression`, `knn`, `decision_tree`, `naive_bayes`
- **Robustos:** `random_forest`, `gradient_boosting`, `svm`

Todos passam por uma `Pipeline` `StandardScaler → estimador`.

## Dataset Iris

- **150 amostras**, 3 classes (`setosa`, `versicolor`, `virginica`), 50 por classe.
- **4 features** em cm: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
"""


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=LONG_DESCRIPTION,
        summary=settings.app_description,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
        contact={"name": "ML FastAPI"},
        license_info={"name": "MIT"},
    )

    # Routers
    app.include_router(health_controller.router)
    app.include_router(algorithm_controller.router)
    app.include_router(training_controller.router)
    app.include_router(prediction_controller.router)
    app.include_router(plot_controller.router)

    # Exception handlers translate domain errors into HTTP responses.
    @app.exception_handler(AlgorithmNotFoundError)
    async def _algorithm_not_found(_: Request, exc: AlgorithmNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ModelNotTrainedError)
    async def _model_not_trained(_: Request, exc: ModelNotTrainedError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(InvalidInputError)
    async def _invalid_input(_: Request, exc: InvalidInputError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    logger.info("Application initialized: %s v%s", settings.app_name, settings.app_version)
    return app


app = create_app()
