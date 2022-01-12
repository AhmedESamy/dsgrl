from loss import SimplifiedRegularizedLoss, ModelRegularizer
from model import Surgeon, LogisticRegression
from augmentor import get_augmentor
from gnn import Encoder
import datasets
import utils


from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from collections import namedtuple
import numpy as np
import torch
import os.path as osp
from tqdm import tqdm

torch.manual_seed(1)


class NodeClassificationExperiment:

    def __init__(self, args):
        self._args = args
        self.__init_data()
        self.__init_state()
        

    def __init_data(self):
        cd = datasets.CompiledDataset(self._args)
        compiled_dataset = cd.compile()
        self.dataset = compiled_dataset[utils.DATASET]
        self._loader = compiled_dataset[utils.TRAIN_LOADER]
        self._subgraph_loader = None
        if utils.SUBGRAPH_LOADER in compiled_dataset:
            self._subgraph_loader = compiled_dataset[utils.SUBGRAPH_LOADER]

    def __init_state(self):
        args = self._args
        self.device = utils.get_device()
        loss_fn = SimplifiedRegularizedLoss(inv_w=args.inv_w, cov_w=args.cov_w)
        mod_reg = ModelRegularizer(mod_w=args.mod_w)
        learner, optimizer = self.__reset_model()
        self.state = {"learner": learner, "optimizer": optimizer, 
                      "loss_fn": loss_fn, "mod_reg": mod_reg}
        self.state = (
            namedtuple("StateDict", self.state.keys())
            (*self.state.values())
        )

    def __reset_model(self):
        print("Initializing model ...")
        args = self._args
        
        aug_in_dim = self.dataset.data.x.shape[1]
        aug_out_dim = args.aug_dim
        enc_in_dim = aug_out_dim if args.aug_type == "feature" else self.dataset.data.x.shape[1]
        enc_out_dim = args.model_dim
        
        augmentor = get_augmentor(in_dim=aug_in_dim,  out_dim=aug_out_dim, conf=args)

        encoder_type = "sage" if args.mini_batch else "gcn"
        net = Encoder(
            in_dim=enc_in_dim, out_dim=enc_out_dim,
            dropout=args.dropout, num_layers=args.num_gnn_layers,
            encoder=encoder_type, skip=True)

        learner = Surgeon(net=net, augmentor=augmentor).to(self.device)
        
        optimizer = torch.optim.Adam(
            learner.parameters(), lr=args.lr)
        return learner, optimizer

    def __train_epoch(self, epoch, epochs):
        learner, optimizer = self.state.learner, self.state.optimizer
        losses = []
        
        torch.autograd.set_detect_anomaly(True)
        for data in self._loader:
            surgeon_input = utils.to_surgeon_input(batch=data, full_data=self.dataset.data)
            z1, z2 = learner(**surgeon_input)
            loss = self.state.loss_fn(z1, z2)
            mod_reg = self.state.mod_reg(learner.augmentor)
            loss = loss + mod_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            msg = f"Epoch: {epoch + 1:03d}/{epochs:03d} training loss: {loss:.4f}"
            utils.log(msg, verbose=self._args.verbose)
            losses.append(float(loss.detach()))
        return np.mean(losses)

    def run(self):
        print("Training ...")
        args = self._args
        learner, optimizer = self.state.learner, self.state.optimizer
        learner.train()
        for epoch in range(args.epochs):
            self.__train_epoch(epoch, args.epochs)

    def resume(self, epochs=1):
        for epoch in range(epochs):
            self.__train_epoch(epoch, epochs)

    def pause_training_mode(self):
        self.state.learner.eval()

    def infer_embedding(self):
        args = self._args
        
        learner = self.state.learner
        learner.eval()
        data = self.dataset.data
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        inference_args = {"x": data.x, "edge_index": data.edge_index, "edge_attr": edge_attr}
        if args.mini_batch:
            inference_args["loader"] = self._subgraph_loader
            
        return learner.infer(**inference_args)
        
    def save_hon(self):
        path = osp.join(self.dataset.processed_dir, "hon.pt")
        torch.save(self.state.learner.hon, path)

    def reset_model(self):
        self.__init_state()


class LinearEvaluationExperiment:

    def __init__(self, in_dim, out_dim, device, task, verbose=True, epochs=100):
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._device = device
        self._task = task
        self._metric = "accuracy" if task in {"bc", "mcc"} else "roc_auc"
        self._verbose = verbose
        self._epochs = epochs

    def __feed(self, x, y, mask=None):
        if mask is None:
            return self._classifier(x, y)
        mask.to(self._device)
        return self._classifier(
            x.to(self._device)[mask],
            y.to(self._device)[mask]
        )

    def score(self, scores, truth):

        if self._metric == "accuracy":
            prediction = torch.argmax(scores, dim=1)
            score = (
                    (prediction.to(self._device) == truth.to(self._device)).sum() /
                    truth.shape[0]
            )
            return score * 100
        elif self._metric == "roc_auc":
            # scores = (scores > 0).float()
            return (
                roc_auc_score(truth.detach().cpu().numpy(), scores.detach().cpu().numpy())
            )

    def execute(self, x, y, train_mask, val_mask=None, test_mask=None):
        if train_mask.ndim == 1:
            train_mask = train_mask.view(-1, 1)
            val_mask = val_mask.view(-1, 1)
            test_mask = test_mask.view(-1, 1)
        num_splits = train_mask.shape[1]
        
        seeds = range(num_splits)
        val_accs, test_accs = [], []
        val_msg = test_msg = ""
        val_best = test_best = 0.
        
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

                if (i + 1) % 5 == 0:
                    self._classifier.eval()
                    train_acc = self.score(scores=logits, truth=y[train_mask].squeeze())
                    val_logits, _ = self.__feed(x=x, y=y, mask=val_mask)
                    val_acc = self.score(scores=val_logits, truth=y[val_mask].squeeze())
                    test_logits, _ = self.__feed(x=x, y=y, mask=test_mask)
                    test_acc = self.score(scores=test_logits, truth=y[test_mask].squeeze())

                    if val_acc > val_best:
                        val_best = val_acc
                        test_best = test_acc
                        
            # train_acc = self.score(scores=logits, truth=y[train_mask].squeeze())
            # val_logits, _ = self.__feed(x=x, y=y, mask=val_mask)
            # val_acc = self.score(scores=val_logits, truth=y[val_mask].squeeze())
            # test_logits, _ = self.__feed(x=x, y=y, mask=test_mask)
            # test_acc = self.score(scores=test_logits, truth=y[test_mask].squeeze())
            val_acc, test_acc = val_best, test_best
            val_msg = f"validation {self._metric}: {val_acc:.2f}"
            test_msg = f"test {self._metric}: {test_acc:.2f}"

            val_accs.append(float(val_acc))
            test_accs.append(float(test_acc))

            msg = (
                f"Split {split_idx + 1:03d}/{num_splits:03d}. Training {self._metric}: "
                f"{train_acc:.2f} {val_msg} {test_msg}".strip()
            )

            utils.log(msg, verbose=self._verbose)

        if len(val_accs) > 0:
            print("Validation", np.mean(val_accs), np.std(val_accs))
        if len(test_accs) > 0:
            print("Test", np.mean(test_accs), np.std(test_accs))

        return np.mean(val_accs)
