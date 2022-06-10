from loss import RegularizedLoss, ModelRegularizer
from dataset.dataset import UnifiedGraphDataset
from augmentors import get_augmentor
from gnn import get_encoder
from utils import Aggregator
from utils import get_device
from train import Trainer
from model import DSGRL

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
try:
    from sklearnex import patch_sklearn, config_context
    patch_sklearn()
    use_gpu = True
except ModuleNotFoundError:
    use_gpu = False
import torch
import numpy as np
import logging
import optuna
import yaml
import sys
import os
import os.path as osp
from tqdm import tqdm


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class GraphPropertyPredictionExperiment:
    
    def __init__(self, conf):
        self.conf = conf
        self._init()
        
    def __objective(self, trial):
        """
        The objective function used by optuna for tuning hyperparameters.
        Returns the validation score for a given trial.
        
        Args:
            trial: The i-th trial object for the tuning experiment from which new
                hyper-parameter configuration will be sampled
        
        """
        self._sample_params(trial)
        self.__reset_loss_fn()
        report = self._run()
        return report["validation"]["average_accuracy"]
    
    def __get_paths(self):
        """
        Returns basic paths for logging training scores and and hyper-parameter
        configurations
        """
        
        data_dir = osp.split(self.dataset.raw_dir)[0]
        log_dir = osp.join(data_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        config_dir = osp.join("params", self.conf.name.lower())
        os.makedirs(config_dir, exist_ok=True)
        
        return {
            "config": osp.join(config_dir, f"{self.conf.aug_type.lower()}.yaml"),
            "loss": osp.join(log_dir, f"{self.conf.aug_type.lower()}_loss_log.txt"),
            "aug_reg": osp.join(log_dir, f"{self.conf.aug_type.lower()}_aug_reg_log.txt")
        }

    def __reset_loss_fn(self):
        """
        Initializes or resets the loss function during training and model-selection.
        """
        conf = self.conf
        self.mod_reg_fn = ModelRegularizer(λ=conf.mod_w)
        self.loss_fn = RegularizedLoss(α=conf.inv_w, β=conf.var_w, γ=conf.cov_w)
        print(self.loss_fn)
        print(self.mod_reg_fn)
        print(f"Reseting loss function with invariance weight: {conf.inv_w} "
              f"and variance weight: {conf.var_w} and "
              f"covariance weight: {conf.cov_w} and "
              f"model regularization weight {conf.mod_w}")
    
    def _sample_params(self, trial):
        """
        Samples hyper-parameter values for during model selection (tuning) experiment
        """
        self.conf.lr = trial.suggest_float("lr", 1e-4, 1e-1)
        self.conf.dropout = trial.suggest_float("dropout", 0.0, 1.0)
        self.conf.num_gnn_layers = trial.suggest_categorical("num_gnn_layers", [1, 2, 3, 4])
        self.conf.aug_dim = trial.suggest_int("aug_dim", 32, 512)
        self.conf.model_dim = trial.suggest_int("model_dim", 32, 64)
        self.conf.inv_w = float(trial.suggest_int("inv_w", 1, 100))
        self.conf.var_w = float(trial.suggest_int("var_w", 1, 100))
        self.conf.cov_w = float(trial.suggest_int("cov_w", 1, 100))
        self.conf.mod_w = float(trial.suggest_int("mod_w", 1, 100))
        self.conf.epochs = 1
        if self.conf.aug_type in {"top", "topology", "t"}:
            self.conf.edge_mul = trial.suggest_int("edge_mul", 2, 10)
            self.conf.keep_edges = trial.suggest_categorical("keep_edges", [True, False])
            
    def _init(self):
        """
        Initializes the basic objects that do are constant throught an experiment.
        
        """
        
        conf = self.conf
        self.__reset_loss_fn()
        self.agg_fn = Aggregator()
        self.dataset = UnifiedGraphDataset(
            root=conf.root, name=conf.name, degree_profile=conf.degree_profile)
        print(self.dataset.data)
        self.splits = self.dataset.splits
        self.loader = DataLoader(
            self.dataset, batch_size=self.conf.batch_size,
            num_workers=32, shuffle=True)
        
    def _run(self):
        """
        Executes training and evaluation experiments
        
        Returns
        -------
        Evaluation report
        """
        conf = self.conf
        device = get_device()
        has_edge_attr = (hasattr(self.dataset.data, "edge_attr") and 
                         self.dataset.data.edge_attr is not None)
        edge_dim = self.dataset.data.edge_attr.shape[1] if has_edge_attr else None

        # Build the augmenter network
        augmentor = get_augmentor(
            num_features=self.dataset.num_features, num_edge_features=edge_dim, conf=self.conf)
        # Build the encoder network
        encoder = get_encoder(conf=self.conf, num_features=self.dataset.num_features)
        # Build the complete DSGRL model
        model = DSGRL(augmentor=augmentor, encoder=encoder).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
        trainer = Trainer(loader=self.loader, model=model, optimizer=optimizer, 
                          loss_fn=self.loss_fn, mod_reg_fn=self.mod_reg_fn,
                          agg_fn=self.agg_fn, epochs=conf.epochs)
        
        if not conf.tune:
            # Evaluate before training the model to see how the random model compares to the trained model.
            self.evaluator = LinearEvaluationExperiment(
                splits=self.splits, report_test=True, verbose=not conf.tune)
            x, y = trainer.infer(self.loader, desc="Inferring embeddings before training")
            
            report = self.evaluator.evaluate(x, y, desc="Linear evaluation before training")
            self.evaluator.log_report(report=report, name=conf.name, desc="Evaluation report before training")
        
        trainer.fit()
        
        x, y = trainer.infer(self.loader, desc="Inferring embeddings after training")
        
        self.evaluator = LinearEvaluationExperiment(
            splits=self.splits, report_test=True, verbose=not conf.tune)
        
        report = self.evaluator.evaluate(x, y, desc="Linear evaluation after training")
        self.evaluator.log_report(report=report, name=conf.name, desc="Evaluation report before training")
        
        paths = self.__get_paths()
        trainer.log_loss(paths["loss"])
        trainer.log_reg(paths["aug_reg"])
        return report
    
    def tune(self):
        """
        Executes a model selection (hyper-parameter tuning) experiment using the validation set
        """
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(self.__objective, n_trials=self.conf.trials)
        
        print("Best trial:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        with open(self.__get_paths()["config"], 'w') as f:
            yaml.safe_dump(trial.params, f)
            
    def run(self):
        if self.conf.tune:
            self.tune()
        else:
            self._run()

    
class LinearEvaluationExperiment:
    
    def __init__(self, splits, task="clf", metrics="accuracy", report_train=True, 
                 report_val=True, report_test=False, verbose=True, estimator="logistic"):
        self.splits = splits
        self.task = task
        self.metrics = metrics
        self.report_train = report_train
        self.report_val = report_val
        self.report_test = report_test
        self.verbose = verbose
        self.estimator = estimator
        self.supported_metrics = {
            "accuracy": accuracy_score,
            "roc_auc": roc_auc_score,
            "mae": mean_absolute_error
        }
        
    def __str__(self):
        args = ",\n\t\t".join(f"{field}={value}" for field, value in self.__dict__.items())
        return f"{type(self).__name__}(\n\t\t{args}\n\t)"
        
    def __repr__(self):
        return str(self)
        
    def evaluate(self, x, y, desc="Linear evaluation"):
        y = y.squeeze().astype("int")
        splits = self.splits
        results = {"train": [], "val": [], "test": []}
        scaler = StandardScaler()
        evaluation_metric = self.supported_metrics[self.metrics]
        for i in tqdm(range(len(splits)), desc=desc):
            model_selection = splits[i]["model_selection"]
            test_indices = splits[i]["test"]
            train_indices = model_selection[0]["train"]
            val_indices = model_selection[0]["validation"]
            
            x_train, y_train = x[train_indices], y[train_indices]
            x_val, y_val = x[val_indices], y[val_indices]
            x_test, y_test = x[test_indices], y[test_indices]

            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

            seeds = [987, 79868, 9080]
            train_scores = []
            val_scores = []
            test_scores = []
            for seed in seeds:       
                if self.estimator == "logistic":
                    clf = LogisticRegression(solver="liblinear", random_state=seed)
                elif self.estimator == "knn":
                    clf = KNeighborsClassifier(n_neighbors=1)
                
                clf.fit(x_train, y_train)

                y_tr = clf.predict(x_train)
                y_v = clf.predict(x_val)
                y_te = clf.predict(x_test)

                
                tr_score = evaluation_metric(y_train, y_tr)
                v_score = evaluation_metric(y_val, y_v)
                te_score = evaluation_metric(y_test, y_te)

                train_scores.append(tr_score)
                val_scores.append(v_score)
                test_scores.append(te_score)

            results["train"].append(np.mean(train_scores) * 100.)
            results["val"].append(np.mean(val_scores) * 100.)
            results["test"].append(np.mean(test_scores) * 100.)

        report = {}
        if self.report_train:
            report["training"] =  {
                f"average_{self.metrics}": np.mean(results["train"]),
                "std": np.std(results["train"])
            }
        if self.report_val:
            report["validation"] = {
                f"average_{self.metrics}": np.mean(results["val"]), 
                "std": np.std(results["val"])
            }
        if self.report_test:
            report["test"] = {
                f"average_{self.metrics}": np.mean(results["test"]), 
                "std": np.std(results["test"])
            }
        return report
    
    def log_report(self, report, name=None, desc="Evaluation report"):
        if self.verbose:
            print()
            print(desc)
            if name is not None:
                print(f"\tDataset: {name}")
            for k, v in report.items():
                print(f"\t\tSplit: {k.title()}")
                for metric, score in v.items():
                    print(f"\t\t\t{metric}: {score}")