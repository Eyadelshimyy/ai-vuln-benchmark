import pytest
import numpy as np

from fast_mlsirm.io import load_factor_csv, load_params

def test_load_factor_csv_empty(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    with pytest.raises(ValueError, match="factor CSV is empty"):
        load_factor_csv(empty_csv)


@pytest.mark.parametrize(
    "payload",
    [
        # Fuzz reproducer: U+109249 segfaulted NumPy's loadtxt tokenizer.
        b"\n,\xf4\x89\x89\x89",
        # U+10000 was silently mis-parsed by loadtxt as the integer 65488.
        "item_id,factor_id\n0,\U00010000".encode("utf-8"),
    ],
    ids=["segfault_reproducer", "garbage_parse_reproducer"],
)
def test_load_factor_csv_rejects_supplementary_plane_characters(tmp_path, payload):
    hostile_csv = tmp_path / "hostile.csv"
    hostile_csv.write_bytes(payload)

    with pytest.raises(ValueError, match="Basic Multilingual Plane"):
        load_factor_csv(hostile_csv)


def test_load_factor_csv_accepts_bmp_header_text(tmp_path):
    labeled_csv = tmp_path / "labeled.csv"
    labeled_csv.write_text(
        "문항_id,요인_id\n0,1\n1,0\n", encoding="utf-8"
    )

    result = load_factor_csv(labeled_csv)

    assert result.tolist() == [1, 0]

def test_load_params(tmp_path):
    params_file = tmp_path / "params.npz"
    np.savez(
        params_file,
        theta=np.zeros((10, 2)),
        alpha=np.zeros(4),
        b=np.zeros(4),
        xi=np.zeros((10, 2)),
        zeta=np.zeros((4, 2)),
        tau=np.array(1.0)
    )

    params = load_params(params_file)
    assert params.theta.shape == (10, 2)
    assert params.tau == 1.0
