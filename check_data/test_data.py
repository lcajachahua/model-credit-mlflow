import scipy.stats
import pandas as pd


def test_column_presence_and_type(data):
    # Disregard the reference dataset
    _, data = data

    required_columns = {
        "LIMIT_BAL": pd.api.types.is_integer_dtype,
        "AGE": pd.api.types.is_integer_dtype,
        "PAY_0": pd.api.types.is_integer_dtype,
        "PAY_2": pd.api.types.is_integer_dtype,
        "PAY_3": pd.api.types.is_integer_dtype,
        "PAY_4": pd.api.types.is_integer_dtype,
        "PAY_5": pd.api.types.is_integer_dtype,
        "PAY_6": pd.api.types.is_integer_dtype,
        "BILL_AMT1": pd.api.types.is_integer_dtype,
        "BILL_AMT2": pd.api.types.is_integer_dtype,
        "BILL_AMT3": pd.api.types.is_integer_dtype,
        "BILL_AMT4": pd.api.types.is_integer_dtype,
        "BILL_AMT5": pd.api.types.is_integer_dtype,
        "BILL_AMT6": pd.api.types.is_integer_dtype,
        "PAY_AMT1": pd.api.types.is_integer_dtype,
        "PAY_AMT2": pd.api.types.is_integer_dtype,
        "PAY_AMT3": pd.api.types.is_integer_dtype,
        "PAY_AMT4": pd.api.types.is_integer_dtype,
        "PAY_AMT5": pd.api.types.is_integer_dtype,
        "PAY_AMT6": pd.api.types.is_integer_dtype,
        "SEX_1": pd.api.types.is_integer_dtype,
        "SEX_2": pd.api.types.is_integer_dtype,
        "EDUCATION_0": pd.api.types.is_integer_dtype,
        "EDUCATION_1": pd.api.types.is_integer_dtype,
        "EDUCATION_2": pd.api.types.is_integer_dtype,
        "EDUCATION_3": pd.api.types.is_integer_dtype,
        "EDUCATION_4": pd.api.types.is_integer_dtype,
        "EDUCATION_5": pd.api.types.is_integer_dtype,
        "EDUCATION_6": pd.api.types.is_integer_dtype,
        "MARRIAGE_0": pd.api.types.is_integer_dtype,
        "MARRIAGE_1": pd.api.types.is_integer_dtype,
        "MARRIAGE_2": pd.api.types.is_integer_dtype,
        "MARRIAGE_3": pd.api.types.is_integer_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):
    # Disregard the reference dataset
    _, data = data

    # Check that only the known classes are present
    known_classes = [
        0,
        1
    ]

    assert data["default"].isin(known_classes).all()


def test_kolmogorov_smirnov(data, ks_alpha):
    sample1, sample2 = data

    columns = ['LIMIT_BAL',
               'AGE',
               'PAY_0',
               'PAY_2',
               'PAY_3',
               'PAY_4',
               'PAY_5',
               'PAY_6',
               'BILL_AMT1',
               'BILL_AMT2',
               'BILL_AMT3',
               'BILL_AMT4',
               'BILL_AMT5',
               'BILL_AMT6',
               'PAY_AMT1',
               'PAY_AMT2',
               'PAY_AMT3',
               'PAY_AMT4',
               'PAY_AMT5',
               'PAY_AMT6',
               'SEX_1',
               'SEX_2',
               'EDUCATION_0',
               'EDUCATION_1',
               'EDUCATION_2',
               'EDUCATION_3',
               'EDUCATION_4',
               'EDUCATION_5',
               'EDUCATION_6',
               'MARRIAGE_0',
               'MARRIAGE_1',
               'MARRIAGE_2',
               'MARRIAGE_3']

    # Bonferroni correction for multiple hypothesis testing
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha_prime = 1 - (1 - ks_alpha) ** (1 / len(columns))

    for col in columns:
        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])

        assert p_value > alpha_prime
