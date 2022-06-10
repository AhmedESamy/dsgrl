import experiment
import utils

import logging
import optuna

import torch

import sys
import os

import yaml

torch.manual_seed(0)

current_best = 0


def tune(args):
    args.device = utils.get_device()
    def save_hyper_params(trial):
        with open(f"./params/{args.name.lower()}/{args.aug_type.lower()}.yaml", 'w') as f:
            yaml.safe_dump(trial.params, f)
            
    def objective(trial=None):
        global current_best
        torch.cuda.empty_cache()
        if trial is not None:
            args.lr = trial.suggest_float("lr", 1e-4, 1e-1)
            args.inv_w = trial.suggest_float("inv_w", 1e-4, 1.0)
            args.cov_w = trial.suggest_float("cov_w", 1e-4, 1.0)
            args.var_w = trial.suggest_float("var_w", 1e-4, 1.0)
            args.mod_w = trial.suggest_float("mod_w", 1e-4, 1.0)
            args.dropout = trial.suggest_float("dropout", 0.1, 0.9)
            args.aug_dim = trial.suggest_int("aug_dim", 64, 512)
            if args.aug_type == "topology":
                args.keep_edges = trial.suggest_categorical("keep_edges", [True, False])
                args.edge_mul = trial.suggest_int("edge_mul", 1, 10)

        exp = experiment.NodeClassificationExperiment(args=args)
        exp.run()
        val_acc = exp.eval_model(test=False)
        if val_acc > current_best:
            current_best = val_acc
            save_hyper_params(trial)
            
        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()
        return val_acc
            
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    
    
        
