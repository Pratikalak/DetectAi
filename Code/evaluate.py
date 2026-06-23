import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             auc, precision_recall_curve, f1_score, accuracy_score,
                             precision_score, recall_score)
from tensorflow.keras.models import load_model

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load model
print("Loading model...")
model = load_model('deepfake_detection_model.h5')

# Load test data
print("Loading test data...")
try:
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
except FileNotFoundError:
    print("❌ Error: 'test_data.pkl' not found. Please run 'train.py' first.")
    exit(1)

print(f"Test set size: {len(X_test)} images")

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(X_test, verbose=0)
y_pred_proba = predictions[:, 1]  # Probability of class 1 (Fake)
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])

# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_curve, precision_curve)

# Print results
print("\n" + "="*60)
print("EVALUATION METRICS (HELD-OUT TEST SET)")
print("="*60)
print(f"\nAccuracy:                   {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision:                  {precision:.4f}")
print(f"Recall (Sensitivity):       {recall:.4f}")
print(f"F1-Score:                   {f1:.4f}")
print(f"ROC-AUC Score:              {roc_auc:.4f}")
print(f"PR-AUC Score:               {pr_auc:.4f}")

print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Real    Fake")
print(f"Actual Real   {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"      Fake   {cm[1,0]:3d}    {cm[1,1]:3d}")

print(f"\nDetailed Classification Report:")
print(report)
print("="*60)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix.png")
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curve.png")
plt.show()

# Confidence distribution plot
plt.figure(figsize=(10, 6))
confidence_real = predictions[y_test == 0, 1]  # Confidence for Real images
confidence_fake = predictions[y_test == 1, 1]  # Confidence for Fake images

plt.hist(confidence_real, bins=20, alpha=0.6, label='Real Images', color='green', edgecolor='black')
plt.hist(confidence_fake, bins=20, alpha=0.6, label='Fake Images', color='red', edgecolor='black')
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
plt.xlabel('Model Confidence (Probability)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Confidence Distribution by Class (Test Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confidence_distribution.png")
plt.show()

# Summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"True Positives (Fake detected as Fake):  {cm[1,1]}")
print(f"True Negatives (Real detected as Real):  {cm[0,0]}")
print(f"False Positives (Real detected as Fake): {cm[0,1]}")
print(f"False Negatives (Fake detected as Real): {cm[1,0]}")
print(f"\nSpecificity (True Negative Rate):        {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
print(f"Sensitivity (True Positive Rate):       {cm[1,1]/(cm[1,1]+cm[1,0]):.4f}")
print(f"{'='*60}\n")
