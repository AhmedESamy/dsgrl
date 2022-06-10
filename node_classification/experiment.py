from loss import SimplifiedRegularizedLoss, RegularizedLoss, ModelRegularizer
from model import Surgeon, LogisticRegression
from augmentor import get_augmentor
from gnn import Encoder, HetroEncoder
import datasets
import utils


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn import linear_model
from torch.utils.data import DataLoader
from collections import namedtuple
import numpy as np
import torch
import os.path as osp
from tqdm import tqdm
import time
import sys
try:
    import wandb
    wandb_installed = True
except:
    wandb_installed = False

torch.manual_seed(1)


class NodeClassificationExperiment:

    def __init__(self, args):
        self._args = args
        self.__init_wandb()
        self.__init_data()
        self.__init_state()
        self.__init_utils()
        
    def __init_wandb(self):
        self.wandb_log = False
        if self._args.wandb and wandb_installed:
            wandb.init(
                project="dsgrl", name=self._args.name, 
                entity=self._args.entity, config=vars(self._args)
            )
            self.wandb_log = True
            
    def __init_data(self):
        self.data_module = datasets.DataModule(self._args)
        self.dataset = self.data_module.dataset
        self._loader = self.data_module.train_loader
        self._subgraph_loader = self.data_module.subgraph_loader

    def __init_state(self):
        args = self._args
        self.device = utils.get_device()
        logger = wandb if self.wandb_log else None
        
        loss_fn = RegularizedLoss(α=args.inv_w, β=args.var_w, γ=args.cov_w, logger=logger)
        # loss_fn = SimplifiedRegularizedLoss(inv_w=args.inv_w, cov_w=args.cov_w)
        print(loss_fn)
        mod_w = args.mod_w if args.aug_type == "feature" else 0.
        mod_reg = ModelRegularizer(mod_w=mod_w)
        learner, optimizer = self.__reset_model()
        self.state = {"learner": learner, "optimizer": optimizer, 
                      "loss_fn": loss_fn, "mod_reg": mod_reg}
        self.state = (
            namedtuple("StateDict", self.state.keys())
            (*self.state.values())
        )
        
    def __init_utils(self):
        self.early_stop = utils.EarlyStop(self._args.patience)
        self.best_score_tracker = utils.BestScoreTracker()

    def __reset_model(self):
        print("Initializing model ...")
        args = self._args
        
        aug_in_dim = self.data_module.num_features
        aug_out_dim = args.aug_dim
        enc_in_dim = aug_out_dim if args.aug_type == "feature" else self.dataset.data.x.shape[1]
        enc_out_dim = args.model_dim
        encoder_type = "sage" if args.mini_batch else "gcn"
        if self.data_module.is_hetro:
            augmentor = get_augmentor(
                in_dim=None,  out_dim=aug_out_dim, conf=args, 
                metadata=self.data_module.metadata
            ).to(self.device)
            data = self._loader[0]
            self._init_lazy_module(augmentor, data)
            net = HetroEncoder(
                in_dim=enc_in_dim, out_dim=enc_out_dim,
                dropout=args.dropout, num_layers=args.num_gnn_layers,
                encoder=encoder_type, metadata=self.data_module.metadata)
        else:
            augmentor = get_augmentor(
                in_dim=aug_in_dim,  out_dim=aug_out_dim, conf=args, 
            )
            net = Encoder(
            in_dim=enc_in_dim, out_dim=enc_out_dim,
            dropout=args.dropout, num_layers=args.num_gnn_layers,
            encoder=encoder_type, skip=True)

        learner = Surgeon(net=net, augmentor=augmentor).to(self.device)
        
        optimizer = torch.optim.Adam(
            learner.parameters(), lr=args.lr)
        return learner, optimizer
    
    def _init_lazy_module(self, module, data):
        utils.log("Initializing a lazy module")
        with torch.no_grad():
            surgeon_input = utils.to_surgeon_input(batch=data, device=self.device)
            module(**surgeon_input)

    def __train_epoch(self, epoch, epochs):
        learner, optimizer = self.state.learner, self.state.optimizer
        
        torch.autograd.set_detect_anomaly(True)
        t_loss = 0
        for data in self._loader:
            surgeon_input = utils.to_surgeon_input(
                batch=data, full_data=self.dataset.data, device=self.device
            )
            z1, z2 = learner(**surgeon_input)
            u_loss = self.state.loss_fn(z1, z2)
            mod_reg = self.state.mod_reg(learner.augmentor)
            loss = u_loss + mod_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.wandb_log:
                wandb.log({"train_loss": loss.item(), "unreg_loss": u_loss.item()})
                
            utils.log(msg="Self-supervised training, epoch:"
                      f" {epoch + 1:03d}/{epochs:03d} "
                      f"training loss: {loss:.4f}", verbose=True, cr=True)
            t_loss +=loss.item()
            
        return t_loss / len(self._loader)
    
    def finish(self):
        if self.wandb_log:
            wandb.finish()

    def run(self):
        args = self._args
        learner, optimizer = self.state.learner, self.state.optimizer
        delta = []
        for epoch in range(args.epochs):
            start = time.time()
            learner.train()
            self.__train_epoch(epoch, args.epochs)
            delta.append(time.time() - start)
            print()
            val_score = self.eval_model(test=True)
            if self.early_stop.stop(val_score, epoch):
                utils.log("Early Stoping!", verbose=True)
                break
                
        delta = delta[1:] if len(delta) > 1 else delta
        utils.log("Average run time per epoch: "
                  f"{np.mean(delta):.6f}, "
                  f"+/- {np.std(delta):.6f}", verbose=True)
            
    def resume(self, epochs=1):
        for epoch in range(epochs):
            self.__train_epoch(epoch, epochs)
            self.eval_model()

    def infer_embedding(self, target_key=None):
        args = self._args
        
        learner = self.state.learner
        learner.eval()
        data = self.dataset.data
        inference_args = utils.to_surgeon_input(
            data, device=self.device
        )
        # if self.data_module.is_hetro:
        #     inference_args = {"x": data.x_dict, "edge_index": data.edge_index_dict, "edge_attr": None}
        # else:
        #     edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        #     inference_args = {"x": data.x, "edge_index": data.edge_index, "edge_attr": edge_attr}
        if args.mini_batch:
            inference_args["loader"] = self._subgraph_loader
            
        embeddings = learner.infer(**inference_args)
        if target_key is None:
            return embeddings
        return embeddings[target_key]
        
    def save_hon(self):
        path = osp.join(self.dataset.processed_dir, "hon.pt")
        torch.save(self.state.learner.hon, path)

    def reset_model(self):
        self.__init_state()
        
    def eval_model(self, split=None, test=True):
        self.state.learner.eval()
        data = self.data_module.get_classification_data()
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        
        if split is not None:
            train_mask = train_mask[:, split]
            val_mask = val_mask[:, split]
            test_mask = test_mask[:, split]
            
        if not test:
            test_mask = val_mask
            
        target_key = self.data_module.labeled_nodetype()
        num_classes = self.data_module.num_classes
        embeddings = self.infer_embedding(target_key)
        y = data.y
        if target_key is not None:
            train_rate = 0.2
            # if final:
            #     train_rates = [0.2, 0.4, 0.6, 0.8]
                
            #evaluator = HetroEval(train_rates=train_rates)
            # embeddings = embeddings[test_mask]#.detach().cpu().numpy()
            # y = data.y[test_mask]#.cpu().numpy()
            # results = evaluator.execute(x, y)
            # return results
            
            # Alternative
            te_idx = test_mask.nonzero().flatten()
            te_idx = te_idx[torch.randperm(te_idx.shape[0])]
            tr_size = int(te_idx.shape[0] * train_rate)
            tr_idx, te_idx = te_idx[:tr_size], te_idx[tr_size:]
            train_mask = torch.zeros(y.shape[0], dtype=torch.bool)
            train_mask[tr_idx] = True
            val_mask = torch.zeros(y.shape[0], dtype=torch.bool)
            val_mask[te_idx] = True
            test_mask = val_mask
            self._args.task = "hetro"
            
        
        if not test:
            test_mask = val_mask
        # print(train_mask.shape, val_mask.shape, test_mask.shape)
        # print(f"Num train: {train_mask.sum()}, num valid: {val_mask.sum()}, num test: {test_mask.sum()}")
        # exit(0)
        evaluator = LinearEvaluationExperiment(
            in_dim=embeddings.shape[1], 
            out_dim=num_classes, 
            device=embeddings.device, 
            task=self._args.task, 
            epochs=100) #epochs = 500 for reddit and yelp.
        self.best_score_tracker.set_metric_to_track(
            evaluator._metric_to_track
        )
        
        val_scores, test_scores = evaluator.execute(
            x=embeddings, y=data.y, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )
        self.best_score_tracker.track(val_scores, test_scores)
        val_score = np.mean(val_scores[evaluator._metric_to_track])
        if self.wandb_log:
            wandb.log({"val_score": val_score})
        if isinstance(val_score, tuple):
            print(val_score)
            return val_score[1]
        return val_score


class LinearEvaluationExperiment:

    def __init__(self, in_dim, out_dim, device, task, eval_step=0, verbose=True, epochs=100):
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._device = device
        self._task = task
        self._verbose = verbose
        self._epochs = epochs
        self.__init_metric()
        self.__set_eval_step(eval_step)
        
    def __init_metric(self):
        if self._task in {"hetro"}:
            self._metric_to_track = "f1macro"
        elif self._task in {"bc", "mcc"}:
            self._metric_to_track = "accuracy"
        else:
            self._metric_to_track = "roc_auc"
            
    def __set_eval_step(self, eval_step):
        if eval_step == 0:
            self._eval_step = 1
        elif isinstance(eval_step, int):
            self._eval_step = eval_step
        elif isinstance(eval_step, float):
            assert 0 < eval_step <= 1
            self._eval_step = int(self._epochs * eval_step)

    def __feed(self, x, y, mask=None):
        if mask is None:
            return self._classifier(x, y)
        mask.to(self._device)
        return self._classifier(
            x.to(self._device)[mask],
            y.to(self._device)[mask]
        )

    def score(self, scores, truth):
        score_dict = {}
        truth = truth.detach().cpu().numpy()
        if self._task in {"bc", "mcc"}:
            prediction = torch.argmax(scores, dim=1).detach().cpu().numpy()
            score_dict["accuracy"] = accuracy_score(truth, prediction)
            # score_dict["accuracy"] = (
            #         (prediction.to(self._device) == truth.to(self._device)).sum() /
            #         truth.shape[0]
            # ).detach().cpu().numpy()
        if self._task in {"hetro"}:
            prediction = torch.argmax(scores, dim=1).detach().cpu().numpy()
            score_dict["f1micro"] = f1_score(truth, prediction, average="micro")
            score_dict["f1macro"] = f1_score(truth, prediction, average="macro")
        elif self._task == "mlc":
            # scores = (scores > 0).float()
            score_dict["roc_auc"] = roc_auc_score(
                truth, 
                scores.detach().cpu().numpy()
            )
        return score_dict

    def execute(self, x, y, train_mask, val_mask=None, test_mask=None):
        
        def format_score_dict(score_dict):
            return " ".join(f"{metric}: {score:.4f}" 
                            for metric, score in score_dict.items())
        
        if train_mask.ndim == 1:
            train_mask = train_mask.view(-1, 1)
            val_mask = val_mask.view(-1, 1)
            if test_mask is not None:
                test_mask = test_mask.view(-1, 1)
            else:
                test_mask = val_mask
        num_splits = train_mask.shape[1]
        
        seeds = range(num_splits)
        val_scores, test_scores = {}, {}
        val_msg = test_msg = ""
        val_best = {self._metric_to_track: 0}
        test_best = {self._metric_to_track: 0}
        
        for split_idx in range(num_splits):
            torch.manual_seed(seeds[split_idx])
            self._classifier = LogisticRegression(
                self._in_dim, self._out_dim, task=self._task).to(self._device)
            opt = torch.optim.Adam(
                self._classifier.parameters(), lr=0.01, weight_decay=0.0)
            mask_index = None if len(train_mask.shape) == 1 else split_idx
            train_mask, val_mask, test_mask = utils.index_mask(
                train_mask, val_mask=val_mask,
                test_mask=test_mask, index=mask_index)

            for i in tqdm(range(self._epochs), desc=f"Training a linear classifier for split {split_idx + 1}"):
                self._classifier.train()
                logits, loss = self.__feed(x=x, y=y, mask=train_mask)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if (i + 1) % self._eval_step == 0:
                    self._classifier.eval()
                    train_score = self.score(scores=logits, truth=y[train_mask].squeeze())
                    val_logits, _ = self.__feed(x=x, y=y, mask=val_mask)
                    val_score = self.score(scores=val_logits, truth=y[val_mask].squeeze())
                    test_logits, _ = self.__feed(x=x, y=y, mask=test_mask)
                    test_score = self.score(scores=test_logits, truth=y[test_mask].squeeze())
                    
                    if val_score[self._metric_to_track] > val_best[self._metric_to_track]:
                        val_best = val_score
                        test_best = test_score
                        
            val_score, test_score = val_best, test_best
            fmt_train_score = format_score_dict(train_score)
            fmt_val_score = format_score_dict(val_score)
            fmt_test_score = format_score_dict(test_score)
            train_msg = f"Training {fmt_train_score}"
            val_msg = f"validation {fmt_val_score}"
            test_msg = f"test {fmt_test_score}"
            for key in val_score:
                if key not in val_scores:
                    val_scores[key] = [val_score[key]]
                    test_scores[key] = [test_score[key]]
                else:
                    val_scores[key].append(val_score[key])
                    test_scores[key].append(test_score[key])

            msg = (
                f"Split {split_idx + 1:03d}/{num_splits:03d}. {train_msg} "
                f"{val_msg} {test_msg}".strip()
            )

            utils.log(msg, verbose=self._verbose)
        
        return val_scores, test_scores


class HetroEval:
    
    def __init__(self, train_rates):
        self.train_rates = train_rates
        
        
    def execute(self, x, y):
        train_rates = self.train_rates
        if isinstance(train_rates, float):
            train_rates = [train_rates]
            
        results = {}
        for rate in train_rates:
            clf = linear_model.LogisticRegression(solver="liblinear", max_iter=500, random_state=1)
            tr_size = int(x.shape[0] * rate)
            tr_x = x[:tr_size]
            tr_y = y[:tr_size]
            te_x = x[tr_size:]
            te_y = y[tr_size:]
            clf.fit(tr_x, tr_y)
            te_pred = clf.predict(te_x)
            mic = f1_score(te_y, te_pred, average="micro")
            mac = f1_score(te_y, te_pred, average="macro")
            results[rate] = mic, mac
            print(f"Rate: {rate} f1-micro: {mic}, f1-macro: {mac}")
        if isinstance(self.train_rates, float):
            return results[self.train_rates][1]
        return results
