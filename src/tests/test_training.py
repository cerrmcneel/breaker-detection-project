import os

import yaml


def test_yaml_config_structure():
    yaml_path = "data.yaml"
    
    # 1. Assert that the file exists
    assert os.path.exists(yaml_path)
    
    # 2. Open and load the yaml file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # 3. Assert that 'train' and 'val' keys exist in the config
    assert 'train' in config
    assert 'val' in config     
    # 4. Assert that 'nc' equals 6
    assert config['nc'] == 6
    
    # 5. Assert that class 4 (index 4) in 'names' is exactly 'OVERSURGE'
    assert config['names'][4] == 'OVERSURGE'

def test_model_initialization():
    # We will pretend that src/model/train.py exists and has a function called init_model()
    # It should load the base "yolo26n.pt" (Nano) model and return it.
    from src.model.train import init_model
    
    model = init_model()
    
    # Assert that the model is not None
    assert model is not None
    
    # Assert that it is indeed a YOLO object (from the ultralytics library)
    from ultralytics import YOLO
    assert isinstance(model, YOLO)


def test_mlflow_tracking_setup(tmp_path):
    import mlflow

    db_path = tmp_path / "test_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    exp = mlflow.set_experiment("Test-Experiment")
    assert exp.name == "Test-Experiment"

    with mlflow.start_run(run_name="test_run") as run:
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("mAP50", 0.85)
        assert run.info.run_id is not None

