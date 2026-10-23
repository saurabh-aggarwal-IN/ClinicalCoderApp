from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from clinical_coder import __version__
from clinical_coder.contracts import (
    CONTRACT_VERSION,
    CodeRequest,
    CodeResponse,
    ContractResponse,
    HarnessInfo,
    HealthResponse,
    contract_payload,
)
from clinical_coder.registry import build_harness
from clinical_coder.service import HarnessService
from clinical_coder.settings import Settings, settings

_STARTED_AT = time.monotonic()


def create_app(
    cfg: Settings | None = None,
    service: HarnessService | None = None,
) -> FastAPI:
    cfg = cfg or settings()
    service = service or HarnessService(build_harness(cfg), cfg)

    app = FastAPI(
        title="Clinical Coding Harness",
        version=__version__,
        description=(
            "One contract, three harness implementations. Synthetic clinical "
            "coding evaluation only; not clinical advice."
        ),
    )

    origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        # The API should always speak the contract, even for programming errors.
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": type(exc).__name__,
                    "retryable": False,
                }
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        hermes_version = None
        runner = getattr(service.harness, "runner", None)
        if runner is not None:
            hermes_version = await runner.version()
        status = "ok"
        if cfg.harness_id == "h1" and runner is not None and hermes_version is None:
            status = "degraded"
        return HealthResponse(
            status=status,
            harness=service.harness.info(
                model=cfg.llm_model,
                hermes_version=hermes_version,
            ),
            contract_version=CONTRACT_VERSION,
            uptime_s=round(time.monotonic() - _STARTED_AT, 3),
            redis_configured=bool(os.getenv("REDIS_URL")),
        )

    @app.get("/v1/harness", response_model=HarnessInfo)
    async def harness_info() -> HarnessInfo:
        return service.harness.info(model=cfg.llm_model)

    @app.get("/v1/contract", response_model=ContractResponse)
    async def contract() -> ContractResponse:
        return contract_payload()

    @app.post("/v1/code", response_model=CodeResponse)
    async def code(request: CodeRequest) -> CodeResponse:
        return await service.handle(request)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "clinical_coder.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        workers=1,
    )


if __name__ == "__main__":
    main()
