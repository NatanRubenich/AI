"""
View layer (presenters).

Translates domain objects (training results, algorithm specs) into
Pydantic response schemas. Keeps controllers thin and ensures the API
shape is decided in one place.
"""

from __future__ import annotations

from app.ml.algorithms import AlgorithmSpec
from app.ml.evaluator import EvaluationReport
from app.ml.trainer import TrainingResult
from app.models.schemas import (
    AlgorithmInfo,
    EvaluationSchema,
    PerClassMetrics,
    Prediction,
    PredictResponse,
    TrainResponse,
)


def present_algorithm(spec: AlgorithmSpec) -> AlgorithmInfo:
    return AlgorithmInfo(
        key=spec.key,
        display_name=spec.display_name,
        category=spec.category,  # type: ignore[arg-type]
        description=spec.description,
    )


def present_evaluation(report: EvaluationReport) -> EvaluationSchema:
    return EvaluationSchema(
        accuracy=report.accuracy,
        precision_macro=report.precision_macro,
        recall_macro=report.recall_macro,
        f1_macro=report.f1_macro,
        cv_mean_accuracy=report.cv_mean_accuracy,
        cv_std_accuracy=report.cv_std_accuracy,
        confusion_matrix=report.confusion_matrix,
        per_class={
            name: PerClassMetrics(**metrics)
            for name, metrics in report.per_class.items()
        },
    )


def present_training(result: TrainingResult, model_id: str) -> TrainResponse:
    return TrainResponse(
        algorithm=present_algorithm(result.spec),
        n_train=result.n_train,
        n_test=result.n_test,
        feature_names=result.feature_names,
        target_names=result.target_names,
        evaluation=present_evaluation(result.report),
        model_id=model_id,
    )


def present_predictions(
    algorithm_key: str, raw: list[dict]
) -> PredictResponse:
    return PredictResponse(
        algorithm=algorithm_key,
        predictions=[Prediction(**item) for item in raw],
    )
