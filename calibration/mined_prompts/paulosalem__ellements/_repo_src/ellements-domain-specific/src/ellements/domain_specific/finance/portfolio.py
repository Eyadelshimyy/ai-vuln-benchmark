"""Reusable portfolio schema models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PortfolioOwner(BaseModel):
    """Portfolio owner metadata."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    country: str | None = None


class PortfolioMetadata(BaseModel):
    """Top-level portfolio metadata."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    as_of: str | None = None
    base_currency: str | None = None
    owner: PortfolioOwner = Field(default_factory=PortfolioOwner)


class PortfolioAccount(BaseModel):
    """Portfolio account metadata."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    account_type: str = ""
    currency: str | None = None
    institution: str | None = None
    account_number: str | None = None


class PortfolioActivityEntry(BaseModel):
    """Position activity event."""

    model_config = ConfigDict(extra="allow")

    date: str | None = None
    type: str = ""
    account_id: str | None = None
    quantity: float | None = None
    price: float | None = None
    amount: float | None = None
    balance: float | None = None
    cost_basis_total: float | None = None
    market_value: float | None = None
    fees: float | None = None
    how: str | None = None
    notes: str | None = None


class PortfolioPosition(BaseModel):
    """Portfolio holding row."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    account_id: str = ""
    asset_class: str = ""
    asset_type: str = ""
    name: str = ""
    ticker: str | None = None
    symbol: str | None = None
    currency: str | None = None
    quantity: float | None = None
    avg_cost: float | None = None
    cost_basis_total: float | None = None
    market_value: float | None = None
    target_weight: float | None = None
    activity: list[PortfolioActivityEntry] = Field(default_factory=list)
    notes: str | None = None


class PortfolioTargets(BaseModel):
    """Portfolio target allocations."""

    model_config = ConfigDict(extra="allow")

    by_asset_class: dict[str, float] = Field(default_factory=dict)


class PortfolioWatchAsset(BaseModel):
    """Watched opportunity row."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    ticker: str | None = None
    symbol: str | None = None
    asset_class: str | None = None
    asset_type: str | None = None
    description: str | None = None
    criteria: str | None = None
    watch_for: str | None = None


class PortfolioWatchlist(BaseModel):
    """Watchlist container."""

    model_config = ConfigDict(extra="allow")

    assets: list[PortfolioWatchAsset] = Field(default_factory=list)


class PortfolioDocument(BaseModel):
    """Root portfolio document schema."""

    model_config = ConfigDict(extra="allow")

    schema_version: str | None = None
    portfolio: PortfolioMetadata = Field(default_factory=PortfolioMetadata)
    accounts: list[PortfolioAccount] = Field(default_factory=list)
    targets: PortfolioTargets = Field(default_factory=PortfolioTargets)
    positions: list[PortfolioPosition] = Field(default_factory=list)
    watchlist: PortfolioWatchlist = Field(default_factory=PortfolioWatchlist)
    watch_assets: list[PortfolioWatchAsset] = Field(default_factory=list)
    market_data: dict[str, Any] = Field(default_factory=dict)
