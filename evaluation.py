# evaluation.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, auc, average_precision_score

def evaluate_model(model, X_test, y_test):
    # Check if the model has `predict_proba` or `decision_function`
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_test_proba = model.decision_function(X_test)
    else:
        y_test_proba = model.predict(X_test).ravel()  # For Keras models or models with no probability methods

    y_pred = (y_test_proba > 0.5).astype(int) if y_test_proba.ndim == 1 else model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    confusion = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (Validation Set)')
    plt.show()

    # ROC Curve and AUC Score
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Validation Set)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    average_precision = average_precision_score(y_test, y_test_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Validation Set)')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


    
