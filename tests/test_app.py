"""Every dashboard page must load without errors through the navigation, with results present and with none."""
import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from p3_modellab import config, runner, sweeps
from p3_modellab.dashboard_bundle import export_run
from p3_modellab.report import build_report
from p3_modellab.tasks import volatility
from test_models_runner import AUDIT, synthetic_panel
from test_report import FIRST, with_meta
from test_sweeps import AUDIT_LONG

APP = Path(__file__).resolve().parents[1] / "app"
VIEWS = ["views/home.py", "views/comparison.py", "views/results.py", "views/custom_test.py",
         "views/misses.py", "views/timeline.py", "views/tweaks.py"]


def point_results_at(monkeypatch, root: Path):
    for name, sub in (("RESULTS_DIR", ""), ("DASHBOARD_DATA_DIR", "dashboard_data"), ("RUNS_DIR", "runs"), ("PANELS_DIR", "panels"), ("AUDIT_DIR", "audit"), ("LOCKS_DIR", "locks")):
        monkeypatch.setattr(config, name, root / sub if sub else root)
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))


def open_page(view: str) -> AppTest:
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    at.run()
    if view != "views/home.py":
        at.switch_page(view)
        at.run()
    return at


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    mp = pytest.MonkeyPatch()
    point_results_at(mp, root)
    panel = with_meta(synthetic_panel(firms=150, nonlinear=True))
    p, _, m = volatility.panel_paths()
    p.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(p, index=False)
    m.write_text(json.dumps({"summary": {}}))
    run = runner.run(panel=panel, audit=AUDIT, test_years=[2004, 2005], progress=lambda *_: None)
    build_report(run, panel=panel, first_listed=FIRST, importance_rows=2000, progress=lambda *_: None)
    runner.run(panel=panel, audit=AUDIT, test_years=[2004, 2005], models=["persistence", "har"], progress=lambda *_: None)  # a run without a report
    long_panel = synthetic_panel(firms=60, start="1995-01", end="2024-11", nonlinear=True)
    for kind in sweeps.KINDS:
        sweeps.run_sweep(kind, demo=True, panel=long_panel, audit=AUDIT_LONG, progress=lambda *_: None)
    yield root
    mp.undo()


@pytest.mark.parametrize("view", VIEWS)
def test_pages_load_with_results(filled, monkeypatch, view):
    point_results_at(monkeypatch, filled)
    at = open_page(view)
    assert not at.exception, at.exception


@pytest.mark.parametrize("view", VIEWS)
def test_pages_load_with_no_results(tmp_path, monkeypatch, view):
    point_results_at(monkeypatch, tmp_path)
    at = open_page(view)
    assert not at.exception, at.exception


def test_sidebar_offers_runs_reports_and_results_choice(filled, monkeypatch):
    point_results_at(monkeypatch, filled)
    at = open_page("views/home.py")
    labels = [b.label for b in at.sidebar.button]
    assert "Start run" in labels
    assert any(s.label == "Results from" for s in at.sidebar.selectbox)
    assert any("Build a report" in m.value for m in at.sidebar.markdown)


def test_results_page_reacts_to_choices(filled, monkeypatch):
    point_results_at(monkeypatch, filled)
    at = open_page("views/results.py")
    cards = lambda: [m.value for m in at.markdown if "p3card" in m.value]
    before = cards()
    at.slider[0].set_value((2005, 2005)).run()
    assert not at.exception and before and cards() != before
    at.multiselect[0].set_value(["har"]).run()  # compare with HAR: head-to-head appears
    assert not at.exception and any("Head to head" in s.value for s in at.subheader)
    at.segmented_control(key="view").set_value("3D").run()
    assert not at.exception


def test_saved_comparison_uses_compact_report_artifacts(filled, monkeypatch):
    point_results_at(monkeypatch, filled)
    at = open_page("views/comparison.py")
    labels = [box.label for box in at.selectbox]
    assert not at.exception and "First model" in labels and "Second model" in labels
    text = " ".join(markdown.value for markdown in at.markdown)
    assert "does not train a model" in " ".join(caption.value for caption in at.caption)
    assert "Head-to-head" in text
    assert any(widget.label == "Models in comparison" for widget in at.multiselect)
    assert any("Compare all saved models" in heading.value for heading in at.subheader)
    assert at.dataframe


def test_dashboard_bundle_excludes_forecasts_and_fitted_models(filled, tmp_path):
    run = next(path.parent for path in (filled / "runs").glob("*/report/report_info.json"))
    # report_info.json -> report -> run
    run = run.parent
    exported = export_run(run, tmp_path)
    assert (exported / "manifest.json").exists()
    assert (exported / "report" / "model_summary.json").exists()
    assert not (exported / "predictions.parquet").exists()
    assert not (exported / "models").exists()
    assert export_run(run, tmp_path) == exported


def test_tweaks_page_opens_every_sweep_kind(filled, monkeypatch):
    """Each sweep kind draws different charts (some twice); every one must open without errors."""
    point_results_at(monkeypatch, filled)
    at = open_page("views/tweaks.py")
    box = lambda: next(s for s in at.selectbox if s.label == "Sweep")
    labels = list(box().options)
    for i, label in enumerate(labels):
        box().select_index(i).run()
        assert not at.exception, (label, at.exception)
    assert len({lab.split("  (")[0] for lab in labels}) == 4


def test_landing_page_has_the_button_and_no_data_source_names(filled, monkeypatch):
    point_results_at(monkeypatch, filled)
    at = open_page("views/home.py")
    assert any("See results" in b.label for b in at.button)
    text = " ".join(m.value for m in at.markdown)
    assert "Stock Swing Forecast" in text and "WRDS" not in text and "CRSP" not in text
