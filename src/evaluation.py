import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_model(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    y_pred_proba: Optional[npt.NDArray] = None,
    class_names: Optional[list] = None
) -> Dict[str, Any]:

    if class_names is None:
        class_names = ['Genuine', 'Spam']
    
    metrics = {}
    
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_score'] = float(f1_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred))
    metrics['recall'] = float(recall_score(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        metrics['average_precision'] = float(average_precision_score(y_true, y_pred_proba))
    
    unique, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = {
        class_names[i]: int(count) for i, count in zip(unique, counts)
    }
    
    logger.info(f"Evaluation complete: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    return metrics


def plot_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    save_path: Optional[str] = None,
    class_names: Optional[list] = None
) -> None:

    if class_names is None:
        class_names = ['Genuine', 'Spam']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: npt.NDArray,
    y_pred_proba: npt.NDArray,
    save_path: Optional[str] = None
) -> None:

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pr_curve(
    y_true: npt.NDArray,
    y_pred_proba: npt.NDArray,
    save_path: Optional[str] = None
) -> None:

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_feature_importance(
    model: Any,
    feature_names: Optional[list] = None,
    n_tfidf_features: Optional[int] = None,
    save_path: Optional[str] = None
) -> Dict[str, float]:

    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return {}
    
    importances = model.feature_importances_
    
    contributions = {}
    if n_tfidf_features is not None:
        tfidf_importance = importances[:n_tfidf_features].sum()
        bert_importance = importances[n_tfidf_features:].sum()
        
        contributions['tfidf'] = float(tfidf_importance)
        contributions['bert'] = float(bert_importance)
        
        logger.info(f"TF-IDF contribution: {tfidf_importance:.2%}")
        logger.info(f"BERT contribution: {bert_importance:.2%}")
    
    top_k = min(20, len(importances))
    top_indices = np.argsort(importances)[-top_k:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), importances[top_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Index')
    plt.title(f'Top {top_k} Most Important Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return contributions
