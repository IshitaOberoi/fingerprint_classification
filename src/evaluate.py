import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, dataloader, class_names, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    safe_class_names = [str(c) for c in class_names]
    labels = list(range(len(safe_class_names)))

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=safe_class_names,
            zero_division=0
        )
    )

    return confusion_matrix(y_true, y_pred, labels=labels)
