"""Shared pytest configuration for the SimulationStacker test suite."""


def pytest_configure(config):
    """Register the custom markers used by this suite.

    Args:
        config (pytest.Config): The pytest configuration object.
    """
    config.addinivalue_line(
        'markers',
        'integration: test requires simulation data on /pscratch and is '
        'skipped automatically when that data is absent.')
