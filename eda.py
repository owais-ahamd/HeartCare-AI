import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load dataset

data = pd.read_csv(
    "data/heart.csv"
)


# Create graph folder

os.makedirs(
    "static/graphs",
    exist_ok=True
)


# -----------------------------------
# 1. Target Distribution
# -----------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=data,
    x="target"
)

plt.title(
    "Heart Disease Target Distribution"
)

plt.xlabel(
    "Target"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/target_distribution.png",
    dpi=300
)

plt.close()


# -----------------------------------
# 2. Age Distribution
# -----------------------------------

plt.figure(
    figsize=(7, 5)
)

sns.histplot(
    data["age"],
    bins=20,
    kde=True
)

plt.title(
    "Age Distribution"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/age_distribution.png",
    dpi=300
)

plt.close()


# -----------------------------------
# 3. Correlation Heatmap
# -----------------------------------

plt.figure(
    figsize=(12, 9)
)

sns.heatmap(
    data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/correlation_heatmap.png",
    dpi=300
)

plt.close()


print(
    "EDA graphs created successfully."
)