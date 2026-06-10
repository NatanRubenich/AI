"""Health / metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar saúde da aplicação",
    description=(
        "**Liveness probe**: confirma que o servidor está rodando e respondendo. "
        "Use para health checks de Kubernetes, load balancers ou monitoramento. "
        "Retorna nome e versão da aplicação."
    ),
    response_description="Status da aplicação + metadata.",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
    )
