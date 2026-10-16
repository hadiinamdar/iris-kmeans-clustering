"""
Iris K-Means Clustering - Streamlit App
Run with:  streamlit run app.py

Optional: place Iris.csv next to this file (columns: SepalLengthCm, SepalWidthCm,
PetalLengthCm, PetalWidthCm, Species). Otherwise sklearn's built-in Iris is used.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

st.set_page_config(page_title="Iris K-Means Clustering", page_icon="🌸", layout="wide")
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Data + modelling helpers
# --------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded=None):
    """Load an uploaded CSV, a local Iris.csv, or sklearn's built-in Iris."""
    if uploaded is not None:
        return pd.read_csv(uploaded), "Uploaded file"
    if os.path.exists("Iris.csv"):
        return pd.read_csv("Iris.csv"), "Iris.csv"
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURES)
    df["Species"] = [iris.target_names[i] for i in iris.target]
    return df, "sklearn built-in Iris"


def fit_kmeans(X_scaled, k, seed=42):
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    labels = model.fit_predict(X_scaled)
    return model, labels


def majority_map(y, labels, k):
    """Map each cluster to its majority species (cluster IDs are arbitrary)."""
    mapped = np.zeros_like(labels)
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            mapped[mask] = mode(y[mask], keepdims=True).mode[0]
    return mapped


@st.cache_data
def evaluate_k_range(X_scaled, y, k_min, k_max, seed):
    rows = []
    for k in range(k_min, k_max + 1):
        model, labels = fit_kmeans(X_scaled, k, seed)
        mapped = majority_map(y, labels, k)
        rows.append(
            {
                "K": k,
                "Inertia": model.inertia_,
                "Silhouette Score": silhouette_score(X_scaled, labels),
                "Accuracy": accuracy_score(y, mapped),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Iris CSV (optional)", type="csv")
df, source = load_data(uploaded_file)

missing = [c for c in FEATURES + ["Species"] if c not in df.columns]
if missing:
    st.error(f"The dataset is missing required columns: {', '.join(missing)}")
    st.stop()

final_k = st.sidebar.slider("Final number of clusters (K)", 2, 6, 3)
k_min, k_max = st.sidebar.slider("K range to compare", 2, 10, (2, 6))
seed = st.sidebar.number_input("Random seed", value=42, step=1)
show_pairplot = st.sidebar.checkbox("Show pairplot (slower)", value=False)

# --------------------------------------------------------------------------
# Prepare data
# --------------------------------------------------------------------------
X = df[FEATURES].copy()
species_cat = df["Species"].astype("category")
y = species_cat.cat.codes.to_numpy()  # used only for evaluation
class_names = list(species_cat.cat.categories)

X_scaled = StandardScaler().fit_transform(X)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("K-Means Clustering on the Iris Dataset")
st.caption(
    f"Data source: {source}. Species labels are used only to evaluate the "
    "clusters afterwards, never to fit K-Means."
)

tab_data, tab_k, tab_final, tab_predict = st.tabs(
    ["Data", "Choosing K", "Final model", "Predict a flower"]
)

# --------------------------------------------------------------------------
# Tab 1: Data / EDA
# --------------------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", df.shape[0])
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Summary statistics")
        st.dataframe(df[FEATURES].describe().T, use_container_width=True)
    with right:
        st.subheader("Species distribution")
        st.dataframe(df["Species"].value_counts().rename("Count"), use_container_width=True)

    if show_pairplot:
        st.subheader("Feature relationships")
        with st.spinner("Drawing pairplot..."):
            cols = FEATURES + ["Species"]
            grid = sns.pairplot(df[cols], hue="Species", diag_kind="hist")
            st.pyplot(grid.figure)
            plt.close(grid.figure)

# --------------------------------------------------------------------------
# Tab 2: Elbow, silhouette, accuracy for a range of K
# --------------------------------------------------------------------------
with tab_k:
    metrics = evaluate_k_range(X_scaled, y, k_min, k_max, int(seed))

    st.subheader("Metrics by K")
    display = metrics.copy()
    display["Accuracy (%)"] = display["Accuracy"] * 100
    st.dataframe(display.round(4), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    ks = metrics["K"].tolist()

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ks, metrics["Inertia"], marker="o")
        ax.set(title="Elbow method", xlabel="K", ylabel="Inertia", xticks=ks)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ks, metrics["Silhouette Score"], marker="o", color="tab:green")
        ax.set(title="Silhouette score", xlabel="K", ylabel="Score", xticks=ks)
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ks, metrics["Accuracy"], marker="o", color="tab:red")
        ax.set(title="Accuracy vs. known species", xlabel="K", ylabel="Accuracy",
               xticks=ks, ylim=(0, 1.05))
        st.pyplot(fig)
        plt.close(fig)

    best_sil = metrics.loc[metrics["Silhouette Score"].idxmax(), "K"]
    st.info(
        f"Highest silhouette score is at K = {int(best_sil)}. K = 3 is used by default "
        "because Iris contains three known species."
    )

# --------------------------------------------------------------------------
# Tab 3: Final model
# --------------------------------------------------------------------------
kmeans, cluster_labels = fit_kmeans(X_scaled, final_k, int(seed))
mapped_labels = majority_map(y, cluster_labels, final_k)

with tab_final:
    accuracy = accuracy_score(y, mapped_labels)
    sil = silhouette_score(X_scaled, cluster_labels)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("K", final_k)
    m2.metric("Accuracy", f"{accuracy * 100:.2f}%")
    m3.metric("Silhouette", f"{sil:.4f}")
    m4.metric("Inertia", f"{kmeans.inertia_:.2f}")

    if final_k != len(class_names):
        st.warning(
            f"The data has {len(class_names)} species but K = {final_k}. "
            "Majority-label accuracy is not directly comparable in this case."
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Clusters (PCA projection)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = cluster_labels
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster",
                        palette="tab10", s=60, ax=ax)
        ax.set_title(f"K-Means clusters (K = {final_k})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, mapped_labels, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted species")
        ax.set_ylabel("Actual species")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Species vs. cluster")
    st.dataframe(pd.crosstab(df["Species"], pd.Series(cluster_labels, name="Cluster")),
                 use_container_width=True)

    result = df.copy()
    result["Cluster"] = cluster_labels
    result["Mapped_Species"] = [class_names[i] for i in mapped_labels]

    st.subheader("Clustered dataset")
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Download clustered data (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="iris_clustered.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# Tab 4: Predict cluster for new measurements
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Assign a new flower to a cluster")
    scaler = StandardScaler().fit(X)

    c1, c2 = st.columns(2)
    sl = c1.slider("Sepal length (cm)", float(X.iloc[:, 0].min()), float(X.iloc[:, 0].max()),
                   float(X.iloc[:, 0].mean()), 0.1)
    sw = c2.slider("Sepal width (cm)", float(X.iloc[:, 1].min()), float(X.iloc[:, 1].max()),
                   float(X.iloc[:, 1].mean()), 0.1)
    pl = c1.slider("Petal length (cm)", float(X.iloc[:, 2].min()), float(X.iloc[:, 2].max()),
                   float(X.iloc[:, 2].mean()), 0.1)
    pw = c2.slider("Petal width (cm)", float(X.iloc[:, 3].min()), float(X.iloc[:, 3].max()),
                   float(X.iloc[:, 3].mean()), 0.1)

    if st.button("Predict cluster"):
        new = scaler.transform(pd.DataFrame([[sl, sw, pl, pw]], columns=FEATURES))
        cluster = int(kmeans.predict(new)[0])
        species_guess = class_names[majority_map(y, cluster_labels, final_k)[cluster_labels == cluster][0]]
        st.success(f"Cluster {cluster}, which mostly contains **{species_guess}**.")