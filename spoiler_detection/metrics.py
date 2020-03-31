import torch
from sklearn import metrics


def get_training_metrics(probs, truth):
    probs, truth = probs.detach(), truth.detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(predicted, truth)

    return {
        f"train_acc": torch.tensor(acc),
    }


def get_validation_metrics(probs, truth):
    probs, truth = probs.detach(), truth.detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(predicted, truth)
    auc = metrics.roc_auc_score(truth, probs[:, 1],)

    return {
        f"avg_val_acc": torch.tensor(acc),
        f"avg_val_auc": torch.tensor(auc),
    }


def get_test_metrics(probs, truth):
    probs, truth = probs.detach(), truth.detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(predicted, truth)
    auc = metrics.roc_auc_score(truth, probs[:, 1],)

    return {
        f"avg_test_acc": torch.tensor(acc),
        f"avg_test_auc": torch.tensor(auc),
    }
