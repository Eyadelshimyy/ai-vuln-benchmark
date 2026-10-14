from fastapi import APIRouter

from app.api.v1.endpoints.ai_status import router as ai_router
from app.api.v1.endpoints.analyses import router as analyses_router
from app.api.v1.endpoints.autonomous import router as autonomous_router
from app.api.v1.endpoints.bounty_campaign import router as bounty_campaign_router
from app.api.v1.endpoints.debugging import router as debugging_router
from app.api.v1.endpoints.discovered_repositories import router as discovered_repos_router
from app.api.v1.endpoints.discovery import router as discovery_router
from app.api.v1.endpoints.github import router as github_router
from app.api.v1.endpoints.hackerone import router as hackerone_router
from app.api.v1.endpoints.projects import router as projects_router
from app.api.v1.endpoints.repair import router as repair_router
from app.api.v1.endpoints.reproduction import router as reproduction_router
from app.api.v1.endpoints.security import router as security_router
from app.api.v1.endpoints.security_agent import router as security_agent_router
from app.api.v1.endpoints.security_testing import router as security_testing_router
from app.api.v1.endpoints.test_generation import router as test_generation_router
from app.api.v1.endpoints.test_runs import router as test_runs_router
from app.api.v1.endpoints.verification import router as verification_router

api_router = APIRouter()
api_router.include_router(projects_router)
api_router.include_router(analyses_router)
api_router.include_router(test_runs_router)
api_router.include_router(debugging_router)
api_router.include_router(test_generation_router)
api_router.include_router(reproduction_router)
api_router.include_router(repair_router)
api_router.include_router(verification_router)
api_router.include_router(github_router)
api_router.include_router(hackerone_router)
# v1.1.0 — Autonomous discovery
api_router.include_router(discovery_router)
api_router.include_router(discovered_repos_router)
api_router.include_router(autonomous_router)
api_router.include_router(ai_router)
api_router.include_router(security_router)
api_router.include_router(security_testing_router)
api_router.include_router(security_agent_router)
# Phase 51 — operational bounty campaigns (Phase 49 orchestrator + Phase 50 engines)
api_router.include_router(bounty_campaign_router)
