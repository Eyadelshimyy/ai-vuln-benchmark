"""Resolved-target authorization for live testing.

hostname → DNS resolution → resolved IPs → network policy → connection

An allowed public hostname must not silently become access to a prohibited
internal/private destination. Local-lab mode is exempt and continues to
target loopback.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.security_testing.lab import is_loopback_or_lab_host
from app.security_testing.target import TargetNormalizer, try_ip

Resolver = Callable[[str], Sequence[str]]


def system_resolver(hostname: str) -> tuple[str, ...]:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return ()
    addresses: list[str] = []
    for info in infos:
        addr = info[4][0]
        if addr not in addresses:
            addresses.append(str(addr))
    return tuple(addresses)


@dataclass(frozen=True)
class ResolvedAddressDecision:
    allowed: bool
    reason: str
    hostname: str
    addresses: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedTargetPolicy:
    """Network policy applied to DNS answers before a live connection."""

    allow_private: bool = False
    allow_loopback: bool = False
    allow_link_local: bool = False
    allow_multicast: bool = False
    allow_unspecified: bool = False
    allow_reserved: bool = False
    blocked_networks: tuple[str, ...] = (
        "169.254.169.254/32",  # cloud metadata
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "0.0.0.0/8",
        "fc00::/7",
        "::1/128",
    )
    lab_exempt: bool = True

    @classmethod
    def live(cls) -> ResolvedTargetPolicy:
        return cls()

    @classmethod
    def lab(cls) -> ResolvedTargetPolicy:
        return cls(
            allow_private=True,
            allow_loopback=True,
            allow_link_local=False,
            lab_exempt=True,
            blocked_networks=("169.254.169.254/32",),
        )


@dataclass
class DnsAuthorizer:
    policy: ResolvedTargetPolicy = field(default_factory=ResolvedTargetPolicy.live)
    resolver: Resolver = system_resolver
    normalizer: TargetNormalizer = field(default_factory=TargetNormalizer)

    def authorize(
        self,
        target: str,
        *,
        lab_mode: bool = False,
        lab_hosts: tuple[str, ...] = (),
    ) -> ResolvedAddressDecision:
        try:
            normalized = self.normalizer.normalize(target)
        except ValueError as exc:
            return ResolvedAddressDecision(False, f"Cannot parse target ({exc})", hostname="")
        hostname = normalized.hostname or normalized.ip or ""
        if not hostname:
            return ResolvedAddressDecision(False, "No hostname to resolve", hostname="")

        if (
            lab_mode
            and self.policy.lab_exempt
            and is_loopback_or_lab_host(hostname, extra=lab_hosts)
        ):
            return ResolvedAddressDecision(
                True, "Lab target exempt from live DNS policy", hostname, addresses=()
            )

        literal = try_ip(hostname)
        if literal is not None:
            addresses: tuple[str, ...] = (str(literal),)
        else:
            addresses = tuple(self.resolver(hostname))
            if not addresses:
                return ResolvedAddressDecision(
                    False, "DNS resolution produced no addresses", hostname
                )

        blocked: list[str] = []
        for address in addresses:
            ok, reason = self._address_allowed(address)
            if not ok:
                blocked.append(f"{address} ({reason})")
        if blocked:
            return ResolvedAddressDecision(
                False,
                "Resolved address is prohibited: " + "; ".join(blocked),
                hostname,
                addresses=addresses,
                blocked=tuple(blocked),
            )
        return ResolvedAddressDecision(
            True, "Resolved addresses permitted", hostname, addresses=addresses
        )

    def _address_allowed(self, address: str) -> tuple[bool, str]:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False, "not an IP address"
        if ip.is_loopback and not self.policy.allow_loopback:
            return False, "loopback"
        if ip.is_private and not self.policy.allow_private:
            return False, "private"
        if ip.is_link_local and not self.policy.allow_link_local:
            return False, "link-local"
        if ip.is_multicast and not self.policy.allow_multicast:
            return False, "multicast"
        if ip.is_unspecified and not self.policy.allow_unspecified:
            return False, "unspecified"
        if ip.is_reserved and not self.policy.allow_reserved:
            return False, "reserved"
        for network in self.policy.blocked_networks:
            try:
                net = ipaddress.ip_network(network, strict=False)
            except ValueError:
                continue
            if ip in net and not (
                (ip.is_loopback and self.policy.allow_loopback)
                or (ip.is_private and self.policy.allow_private)
            ):
                return False, f"blocked-network {network}"
            if ip in net and str(net) in {"169.254.169.254/32"}:
                return False, "cloud-metadata"
        if str(ip) == "169.254.169.254":
            return False, "cloud-metadata"
        return True, "ok"


class MappingResolver:
    """Deterministic resolver for tests. Never talks to real DNS."""

    def __init__(self, mapping: dict[str, Sequence[str]]) -> None:
        self.mapping = {key.lower().rstrip("."): tuple(value) for key, value in mapping.items()}

    def __call__(self, hostname: str) -> tuple[str, ...]:
        return self.mapping.get(hostname.lower().rstrip("."), ())
