"""Basic tests for mlops_timeseries package."""

from mlops_timeseries import __version__


def test_version():
    """Test version is not None."""
    assert __version__ is not None


def test_import():
    """Test basic imports work correctly."""
    import mlops_timeseries

    assert hasattr(mlops_timeseries, "__version__")
