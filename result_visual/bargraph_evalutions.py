import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON file
with open("E:\\4thYearProjectCoding\\outputs\\results\\cv_pretrain_summary.json", "r") as f:
    data = json.load(f)

# Extract fold-wise metrics
folds = data["per_fold"]

accuracy = [fold["accuracy"] * 100 for fold in folds]
precision = [fold["precision"] * 100 for fold in folds]
recall = [fold["recall"] * 100 for fold in folds]
specificity = [fold["specificity_macro"] * 100 for fold in folds]
f1 = [fold["f1_macro"] * 100 for fold in folds]

# Compute overall averages
avg_accuracy = np.mean(accuracy)
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)
avg_specificity = np.mean(specificity)
avg_f1 = np.mean(f1)

# Add average as final bar
accuracy.append(avg_accuracy)
precision.append(avg_precision)
recall.append(avg_recall)
specificity.append(avg_specificity)
f1.append(avg_f1)

# Labels
labels = [f"Fold {i+1}" for i in range(len(folds))]
labels.append("Average")

# Bar positions
x = np.arange(len(labels))
width = 0.15

# Create figure
plt.figure(figsize=(14, 7))

# Plot bars
plt.bar(x - 2*width, accuracy, width, label='Accuracy')
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, specificity, width, label='Specificity')
plt.bar(x + width, recall, width, label='Recall')
plt.bar(x + 2*width, f1, width, label='F1 Score')

# Labels and styling
plt.xticks(x, labels, fontsize=11)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Cross Validation Folds", fontsize=12)
plt.title("Performance Metrics Across Folds and Overall Average", fontsize=14)
plt.ylim(95, 100.5)

# Show values on top
for i, vals in enumerate([accuracy, precision, specificity, recall, f1]):
    offset = (-2 + i) * width
    for j, v in enumerate(vals):
        plt.text(
            x[j] + offset,
            v + 0.05,
            f"{v:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90
        )

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# Save figure
plt.savefig("E:\\4thYearProjectCoding\\outputs\\results\\fold_metrics_bargraph.png", dpi=300, bbox_inches='tight')

# Show plot
plt.show()