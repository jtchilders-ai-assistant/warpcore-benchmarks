import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
VIZ = REPO / "viz"
sys.path.insert(0, str(VIZ))

import common  # noqa: E402


def test_every_throughput_model_has_figure_metadata():
    models = {
        path.parents[2].name
        for path in (REPO / "results").glob("*/raw/throughput_sweep/*.log")
        if "CONTAMINATED" not in path.name
    }

    assert models <= common.C.keys()
    assert models <= common.SHORT.keys()
