[mlops_architecture]:_md_img/mlops_architecture.png
[model_serving]:_md_img/serve.png
[drift_dag]:_md_img/drift_dag.png
[drift_dashboard]:_md_img/drift_dashboard.png
[cicd_ml]:_md_img/cicd_ml.png

# Credit Card Default Example

## Used MLOPS Architecture

![alt][mlops_architecture]

### Key points

For sake of simplicity, the components of the architecture will be explain as "should be" followed by "current implementation" (limited scoped, just for the demo)
1. Data Analysis: Should be extracted from the feature store, in this case just a simple csv dataset (credit card default) will be processed [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#]
2. Model tracking/experimentation: Should be a specific in-house instance of Mlflow, in this case we spin-up a Mlflow docker container. The use of model tracking is detailed as a serious of MLFlowProjects (a ds-component with their own dependencies)
3. Source code / repository: In this demo is Github
4. CI/CD Stage: It is done with Github Actions, you can see this in the "Actions" tab on Github
5. Automated Pipelines: It is implemented as an Inference Pipeline (containing preprocess and the ml-model). All the steps is done by distincts MlflowProjects that shows all the machine learnig development lifecycle
6. Model Registry: This is done through "MlFlow model registry", just a model registration (versioned)
7. Trained Model / Model Serving: There is various options for implementing it, could be FlaskApi/FastApi/(Batch approach) but in this demo we support this feature with "Mlflow model serving"
8. ML Prediction Service: By model serving stage, it is an active service listening in real time for post requests (see an example at "serve/" folder). It could reuse features from feature store and additional data/features coming from the request
9. Performance Monitoring: It should be done by integrating monitoring tools used in the company at the time (grafana, splunk, so on), in this demo it is done by evidently-ai orchestrated by airflow thus generating drift metrics (see "drift/" folder and dashboard drift report at "drift/evidently_reports/credit_card_default_data_drift_by_airflow.html")
10. Alert triggers: Should be done by integrating alerting tools the company use at the time (PagerDuty / Slack )
11. Retraining: Should reuse the Automated Pipeline we've created after CI/CD, also triggered by a drift detection rule


#### Starting the Demo

- List images (including hidden)

      docker image ls -a

- List Containers (including hidden)

      docker ps -a

- Move to the Project Repo

      cd Documents/ProjectRepos

- Run the postgress container: 

      docker run --network cesar_net --expose=5432 -p 5432:5432 -d -v $PWD/pg_data_1/:/var/lib/postgresql/data/ --name pg_mlflow -e POSTGRES_USER='user_pg' -e POSTGRES_PASSWORD='pass_pg' postgres

- Download this Repo

      git clone https://github.com/lcajachahua/model-credit-mlflow.git

- After downloading the repo, move to the root folder

      cd model-credit-mlflow

- Build the project

      docker build -t model-credit-mlflow .

- Run the mlflow server container: 

      docker run -d -p 7755:5000 -v $PWD/container_artifacts:$PWD/container_artifacts --env-file local.env --network cesar_net --name test mlflow_cesar

- Activate the Virtual Environment

      conda activate pipeline_test


#### Step MLOps - MLFlow

1. Downloading

       mlflow run ./download -P step=download_data -P file_url="https://github.com/lcajachahua/model-credit-mlflow/raw/main/_data/default_of_credit_card_clients.csv?raw=true" -P artifact_name=raw_data.csv -P artifact_description="Pipeline for data downloading" --experiment-name credit_card_default --run-name download_data
    
2. Preprocessing

       mlflow run ./preprocess -P step=preprocess -P input_step=download_data -P input_artifact=raw_data.csv -P artifact_name=preprocessed_data.csv -P artifact_description="Pipeline for data preprocessing" --experiment-name credit_card_default --run-name preprocess
 
3. Check/tests

       mlflow run ./check_data -P step=check_data -P input_step=preprocess -P reference_artifact=preprocessed_data.csv -P sample_artifact=preprocessed_data.csv -P ks_alpha=0.05 --experiment-name credit_card_default --run-name check_data
    
4. Segregation

       mlflow run ./segregate -P step=segregate -P input_step=preprocess -P input_artifact=preprocessed_data.csv -P artifact_root=data -P test_size=0.3 -P stratify=default --experiment-name credit_card_default --run-name segregate
    
5. Modeling

       mlflow run ./random_forest -P step=random_forest -P input_step=segregate -P train_data=data/data_train.csv -P model_config=rf_config.yaml -P export_artifact=model_export -P random_seed=42 -P val_size=0.3 -P stratify=default --experiment-name credit_card_default --run-name random_forest
    
6. Evaluate

       mlflow run ./evaluate -P step=evaluate -P input_model_step=random_forest -P model_export=model_export -P input_data_step=segregate -P test_data=data/data_test.csv --experiment-name credit_card_default --run-name evaluate

7. See the Online Environment. To finish the UI, press Ctrl+C

        mlflow ui



#### Mlflow Deployment

##### Batch

- Download the mlflow model

      mlflow artifacts download -r <ID DEL MODELO>
        
        
##### Online (Requiere servicio web activo)

    mlflow models serve -m <PATH EXPORTADO>model_export
    

#### Finishing the Environment

- Deactivate the Virtual Environment and finish the docker containers

      conda deactivate
      docker stop test
      docker stop pg_mlflow






![alt][model_serving]




#### Drift detection

![alt][drift_dag]

![alt][drift_dashboard]

#### CI/CD: Continuous machine learning integration

You can see the CI/CD pipeline for the credit card default model with Github Actions ("Actions" tab on Github), each commit triggers and executes the CI/CD pipeline (the last commit https://github.com/lcajachahua/model-credit-mlflow/commit/ff393ce2784151e3ac78f12c215b64bf527697f9#comments)

![alt][cicd_ml]
        

