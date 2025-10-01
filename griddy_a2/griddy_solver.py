import time
import copy
import pathlib
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def to_bfloat16(tensor):
    return tensor.to(torch.bfloat16)

def to_float16(tensor):
    return tensor.to(torch.float16)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GriddySolver(object):
    def __init__(self, **kwargs):
        self.path_prefix = kwargs.pop("path_prefix", ".")
        self.imbalance = kwargs.pop("imbalance", "regular")
        self.batch_size = kwargs.pop("batch_size", 128)
        self.model_type = kwargs.pop("model", "GriddyModel")
        self.device = kwargs.pop("device", "cpu")
        self.loss_type = kwargs.pop("loss_type", "CE")
        self.lr = kwargs.pop("learning_rate", 0.0001)
        self.momentum = kwargs.pop("momentum", 0.9)
        self.reg = kwargs.pop("reg", 0.0005)
        self.beta = kwargs.pop("beta", 0.9999)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.warmup = kwargs.pop("warmup", 0)
        self.print = kwargs.pop("print", True)
        self.optimizer = kwargs.pop("optimizer", "SGD")
        self.betas = kwargs.pop("betas", [0.9, 0.999])
        self.dtype = kwargs.pop("dtype", "default")
        self.early_stop = kwargs.pop("early_stop", False)
        self.max_attempts = kwargs.pop("max_attempts", 10)
        self.download = kwargs.pop("download", True)
        
        self.train_dataset = kwargs.pop("train_dataset")
        self.test_dataset = kwargs.pop("test_dataset")
        self.model_class = kwargs.pop("model_class")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.test_dataset, batch_size=100, shuffle=False
        )

        self.model = self.model_class(**kwargs)

        if self.dtype == 'float16':
            self.model = self.model.half()
        elif self.dtype == 'bfloat16':
            self.model = self.model.bfloat16()

        print(self.model)
        self.model = self.model.to(self.device)

        if self.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        else:
            pass # add other losses

        self.criterion.to(self.device)

        if self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.reg,
            )
        elif self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        self._reset()

    def _reset(self):
        self.best = 0.0
        self.best_cm = None
        self.best_model = None

    def train(self):
        no_improve = 0
        best_loss = float('inf')
        stop_epoch = 0
        train_acc = []
        val_acc = []
        for epoch in range(self.epochs):
            stop_epoch = epoch
            print(f"EPOCH: {epoch}")
            self._adjust_learning_rate(epoch)

            # train loop
            t_acc = self._train_step(epoch)
            train_acc.append(t_acc.item())
            # validation loop
            acc, cm, loss = self._evaluate(epoch)
            val_acc.append(acc.item())
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1

            if acc > self.best:
                self.best = acc
                self.best_cm = cm
                self.best_model = copy.deepcopy(self.model)
            
            if self.early_stop and no_improve >= self.max_attempts:
                print("EARLY STOP E:{} L:{:.4f}".format(epoch, best_loss))
                break

        per_cls_acc = self.best_cm.diag().detach().numpy().tolist()
        if self.print:
            print("Best Prec @1 Acccuracy: {:.4f}".format(self.best))
            for i, acc_i in enumerate(per_cls_acc):
                print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
        
        return self.best.item(), self.best_model, stop_epoch, train_acc, val_acc, per_cls_acc

    def _train_step(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        self.model.train()

        for idx, (data, target) in enumerate(self.train_loader):
            start = time.time()

            data = data.to(self.device)
            target = target.to(self.device)

            out, loss = self._compute_loss_update_params(data, target)

            batch_acc = self._check_accuracy(out, target)

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if self.print and idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.train_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        return acc.avg

    def _evaluate(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        num_class = 10
        cm = torch.zeros(num_class, num_class)
        self.model.eval()

        # evaluation loop
        for idx, (data, target) in enumerate(self.val_loader):
            start = time.time()

            data = data.to(self.device)
            target = target.to(self.device)

            out, loss = self._compute_loss_update_params(data, target)

            batch_acc = self._check_accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if self.print and idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.val_loader),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        if self.print:
            for i, acc_i in enumerate(per_cls_acc):
                print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

        print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm, losses.avg

    def _check_accuracy(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]

        _, pred = torch.max(output, dim=-1)

        correct = pred.eq(target).sum() * 1.0

        acc = correct / batch_size

        return acc

    def _compute_loss_update_params(self, data, target):
        output = None
        loss = None
        if self.model.training:
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
        return output, loss

    def _adjust_learning_rate(self, epoch):
        epoch += 1
        if epoch <= self.warmup:
            lr = self.lr * epoch / self.warmup
        elif epoch > self.steps[2]:
            lr = self.lr * 0.2
        elif epoch > self.steps[1]:
            lr = self.lr * 0.3
        elif epoch > self.steps[0]:
            lr = self.lr * 0.5
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr