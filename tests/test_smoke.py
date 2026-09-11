def test_package_imports_and_exposes_version() -> None:
    import rag

    assert isinstance(rag.__version__, str)
    assert rag.__version__ != ""
