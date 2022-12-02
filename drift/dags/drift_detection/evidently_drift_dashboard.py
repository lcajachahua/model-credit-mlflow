try:
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator

    from datetime import datetime, timedelta
    import pandas as pd
    from sklearn import datasets
    import os

    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab

    from evidently.model_profile import Profile
    from evidently.model_profile.sections import DataDriftProfileSection

    from evidently.pipeline.column_mapping import ColumnMapping

except Exception as e:
    print("Error  {} ".format(e))

dir_path = "reports"
file_path = "credit_card_default_data_drift_by_airflow.html"


def load_data_execute(**context):
    data_frame = pd.read_csv("/usr/local/airflow/datasets/credit_card_default.csv", index_col=0)

    data_columns = ColumnMapping()
    data_columns.numerical_features = ['LIMIT_BAL',
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
                                       'MARRIAGE_3'
                                       ]

    context["ti"].xcom_push(key="data_frame", value=data_frame)
    context["ti"].xcom_push(key="data_columns", value=data_columns)


def drift_analysis_execute(**context):
    data = context.get("ti").xcom_pull(key="data_frame")
    data_columns = context.get("ti").xcom_pull(key="data_columns")

    credit_card_default_data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    credit_card_default_data_drift_dashboard.calculate(data[:200], data[200:], column_mapping=data_columns)

    try:
        os.mkdir(dir_path)
    except OSError:
        print("Creation of the directory {} failed".format(dir_path))

    credit_card_default_data_drift_dashboard.save(os.path.join(dir_path, file_path))


with DAG(
        dag_id="drift_dashboard",
        # schedule_interval="@daily",
        schedule_interval="@once",
        default_args={
            "owner": "airflow",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
            "start_date": datetime(2021, 1, 1),
        },
        catchup=False,
) as f:
    load_data_execute = PythonOperator(
        task_id="load_data_execute",
        python_callable=load_data_execute,
        provide_context=True,
        op_kwargs={"parameter_variable": "parameter_value"},  # not used now, may be used to specify data
    )

    drift_analysis_execute = PythonOperator(
        task_id="drift_analysis_execute",
        python_callable=drift_analysis_execute,
        provide_context=True,
    )

load_data_execute >> drift_analysis_execute
