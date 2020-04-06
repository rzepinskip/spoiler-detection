import torch
from sklearn import metrics


def get_training_metrics(probs, truth):
    probs, truth = probs.cpu().detach(), truth.cpu().detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(truth, predicted)

    return {
        f"train_acc": torch.tensor(acc),
    }


def get_metrics(probs, truth, prefix):
    probs, truth = probs.cpu().detach(), truth.cpu().detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(truth, predicted)
    if truth.min() != truth.max():
        auc = metrics.roc_auc_score(truth, probs[:, 1],)
    else:
        auc = float("nan")

    return {
        f"avg_{prefix}_acc": torch.tensor(acc),
        f"avg_{prefix}_auc": torch.tensor(auc),
        f"{prefix}_pred_pos": sum(predicted == 1) / float(len(predicted)),
        f"{prefix}_true_pos": sum(truth == 1) / float(len(predicted)),
    }


def get_validation_metrics(probs, truth):
    return get_metrics(probs, truth, "val")


def get_test_metrics(probs, truth):
    return get_metrics(probs, truth, "test")
