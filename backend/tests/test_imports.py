def test_import_stock_mlops():
    import stock_mlops  # noqa: F401


def test_import_new_modules():
    from stock_mlops import training, validation, drift, scalers  # noqa: F401
