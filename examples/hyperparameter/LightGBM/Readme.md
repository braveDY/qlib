# LightGBM超参数调优

## Alpha158数据集
第一个终端执行：
```
optuna create‑study --study LGBM_158 --storage sqlite:///db.sqlite3
optuna‑dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```

第二个终端执行：
```
python hyperparameter_158.py
```

## Alpha360数据集
第一个终端执行：
```
optuna create‑study --study LGBM_360 --storage sqlite:///db.sqlite3
optuna‑dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```

第二个终端执行：
```
python hyperparameter_360.py
```