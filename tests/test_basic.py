from mlops_timeseries import __version__

def test_version():
    '''Test version is not None'''
    assert __version__ is not None

def test_import():
    '''Test basic imports'''
    import mlops_timeseries
    assert True
