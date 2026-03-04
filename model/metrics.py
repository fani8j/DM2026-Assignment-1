import numpy as np
from model.utils import onehot_array
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt

def MSE(y,y_pred):
	return np.mean((y_pred -y)**2)
def logloss(y,y_pred):
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
	return np.mean(-np.log(y_pred)*y - np.log(1-y_pred)*(1-y))
def multi_logloss(y,y_pred):
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
	y_onehot = onehot_array(y,y_pred.shape[1])
	return -np.mean(np.log(np.sum(y_onehot * y_pred,axis=1)))
def accuracy(y,y_pred):
	return np.mean(y==y_pred)



def evaluate_binary_classifier(y_true, y_pred, y_proba, title='Model Evaluation'):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    y_proba = np.asarray(y_proba).ravel()

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
    }

    print(title)
    for name, value in metrics.items():
        print(f'{name:>10}: {value:.4f}')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))

    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(fpr, tpr, label=f"AUC={metrics['AUC-ROC']:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.7)
    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    return metrics