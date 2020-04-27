import torch
from sklearn import metrics


def get_accuracy(probs, truth):
    probs, truth = probs.cpu().detach(), truth.cpu().detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(truth, predicted)

    return torch.tensor(acc)


def get_metrics(probs, truth, prefix):
    probs, truth = probs.cpu().detach(), truth.cpu().detach()
    predicted = torch.argmax(probs, dim=1)
    acc = metrics.accuracy_score(truth, predicted)
    f1 = metrics.f1_score(truth, predicted, average="binary")
    if truth.min() != truth.max():
        auc = metrics.roc_auc_score(truth, probs[:, 1],)
        precision, recall, _ = metrics.precision_recall_curve(truth, probs[:, 1],)
        pr_auc = metrics.auc(recall, precision)
    else:
        auc = float("nan")
        pr_auc = float("nan")

    return {
        f"avg_{prefix}_auc": torch.tensor(auc),
        f"avg_{prefix}_acc": torch.tensor(acc),
        f"avg_{prefix}_pr-auc": torch.tensor(pr_auc),
        f"avg_{prefix}_f1": torch.tensor(f1),
        f"{prefix}_pred_pos": sum(predicted == 1) / float(len(predicted)),
        f"{prefix}_true_pos": sum(truth == 1) / float(len(predicted)),
    }


def get_validation_metrics(probs, truth):
    return get_metrics(probs, truth, "val")


def get_test_metrics(probs, truth):
    return get_metrics(probs, truth, "test")
