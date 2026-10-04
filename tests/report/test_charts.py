from pathlib import Path

from harness.metrics.calibration import reliability_bins
from harness.report.charts import plot_reliability_diagram


def test_plot_reliability_diagram_writes_a_png(tmp_path: Path) -> None:
    bins = reliability_bins([0.9, 0.6], [True, False])
    output_path = tmp_path / "charts" / "example.png"
    plot_reliability_diagram(bins, title="test", output_path=output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
