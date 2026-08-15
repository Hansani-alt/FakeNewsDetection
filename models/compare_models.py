import matplotlib.pyplot as plt

# Model accuracies
models = ["Random Forest", "SVM"]
accuracies = [99.85, 99.27]

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

plt.title("Fake News Detection - Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.ylim(95, 100)

# Display accuracy values
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        accuracy + 0.05,
        f"{accuracy:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

# Save graph
plt.savefig(
    "screenshots/model_comparison.png",
    dpi=300
)

plt.show()

print("Model comparison graph saved successfully!")