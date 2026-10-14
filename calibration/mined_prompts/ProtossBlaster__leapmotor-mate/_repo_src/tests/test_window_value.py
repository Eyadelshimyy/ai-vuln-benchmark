"""Window position % → model-native cmd-230 value (#62). The UI slider is a uniform 0–100%; each
model has its own native range for cmd 230 — the LEAP-platform cars (B10, C10, B05) use 0–10 (10 =
fully open; >10 is ignored, confirmed on-car: B10 by us, C10 by kerniger/leapmotor-ha, B05 follows
the B10 platform), the older T03 uses 0–100. Pure command_client (no fastapi) → runs in CI."""
import command_client


class _V:
    def __init__(self, car_type):
        self.car_type = car_type


def _native(monkeypatch, car_type, pct):
    monkeypatch.setattr(command_client._session, "_vehicle", _V(car_type), raising=False)
    return command_client._windows_native(pct)


def test_b10_scale_0_10(monkeypatch):
    # 0–100% maps onto the B10's 0–10 native range (20% vent → 2, full → 10).
    assert [_native(monkeypatch, "B10", p) for p in (0, 20, 50, 100)] == ["0", "2", "5", "10"]


def test_t03_scale_0_100(monkeypatch):
    # the T03 native range is already 0–100 → 1:1.
    assert [_native(monkeypatch, "T03", p) for p in (0, 20, 50, 100)] == ["0", "20", "50", "100"]


def test_c10_and_b05_scale_0_10(monkeypatch):
    # C10 confirmed 0–10 on-car (kerniger / leapmotor-ha); B05 shares the B10 platform → same scale.
    assert [_native(monkeypatch, "C10", p) for p in (0, 20, 50, 100)] == ["0", "2", "5", "10"]
    assert [_native(monkeypatch, "B05", p) for p in (0, 20, 50, 100)] == ["0", "2", "5", "10"]


def test_unknown_model_defaults_to_0_100(monkeypatch):
    # an unmapped car_type falls back to the 1:1 0–100 range (the T03's native scale).
    assert _native(monkeypatch, "ZX9", 100) == "100"


def test_pct_is_clamped(monkeypatch):
    assert _native(monkeypatch, "T03", 150) == "100"
    assert _native(monkeypatch, "T03", -5) == "0"
