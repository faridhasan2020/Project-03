"""
Mall Customer Segmentation & Streamlit Application
K-Means clustering on Annual Income and Spending Score.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from kneed import KneeLocator
except ImportError:  # optional extension, app still works without it
    KneeLocator = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "Mall_Customers.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
K_RANGE = range(2, 11)  # K = 2 ... 10
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data and model functions
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Some copies of the dataset name the column "Genre" instead of "Gender"
    return df.rename(columns={"Genre": "Gender"})


@st.cache_data
def scale_features(df: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return X_scaled, scaler


@st.cache_data
def compute_elbow(X_scaled) -> pd.DataFrame:
    """Inertia for K = 2..10 (does not depend on the slider, so caching is safe)."""
    inertias = [
        KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled).inertia_
        for k in K_RANGE
    ]
    return pd.DataFrame({"K": list(K_RANGE), "Inertia": inertias})


@st.cache_data
def run_kmeans(X_scaled, k: int):
    """K is an input, so each K value gets its own cached result."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, model.cluster_centers_, silhouette_score(X_scaled, labels)


def suggest_k(elbow_df: pd.DataFrame):
    if KneeLocator is None:
        return None
    knee = KneeLocator(
        elbow_df["K"], elbow_df["Inertia"], curve="convex", direction="decreasing"
    )
    return int(knee.knee) if knee.knee is not None else None


# ---------------------------------------------------------------------------
# Load and prepare
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Place Mall_Customers.csv in the data/ folder.")
    st.stop()

df = load_data(DATA_PATH)
X_scaled, scaler = scale_features(df)
elbow_df = compute_elbow(X_scaled)
suggested_k = suggest_k(elbow_df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Clustering Settings")
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)
if suggested_k:
    st.sidebar.info(f"💡 Elbow (KneeLocator) suggests **K = {suggested_k}**")
show_centroids = st.sidebar.checkbox("Show centroids on scatter plot", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Features used:**\n- Annual Income (k$)\n- Spending Score (1-100)\n\n"
    "Both are scaled with `StandardScaler` before K-Means."
)

# ---------------------------------------------------------------------------
# Run K-Means for the selected K
# ---------------------------------------------------------------------------
labels, centers_scaled, sil_score = run_kmeans(X_scaled, k)

clustered = df.copy()
clustered["Cluster"] = labels

# Centroids back in original units for plotting
centers = pd.DataFrame(scaler.inverse_transform(centers_scaled), columns=FEATURES)

# ---------------------------------------------------------------------------
# Title and introduction
# ---------------------------------------------------------------------------
st.title("🛍️ Mall Customer Segmentation")
st.markdown(
    "This application groups mall customers into segments using **K-Means clustering** "
    "on their **Annual Income** and **Spending Score**. Use the **Elbow Method** chart to "
    "judge a reasonable number of clusters, then choose **K** with the slider in the sidebar. "
    "The clusters and charts update automatically."
)

# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------
st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Duplicate Rows", int(df.duplicated().sum()))

tab_preview, tab_stats, tab_types = st.tabs(["Preview", "Summary Statistics", "Column Types"])
with tab_preview:
    st.dataframe(df.head(10), hide_index=True)
with tab_stats:
    st.dataframe(df.describe().round(2))
with tab_types:
    st.dataframe(
        pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values}),
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Elbow Method + Cluster Visualization (side by side)
# ---------------------------------------------------------------------------
st.header("2. Choosing K and Viewing Clusters")
left, right = st.columns(2)

with left:
    st.subheader("Elbow Method")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(elbow_df["K"], elbow_df["Inertia"], marker="o", color="#1f77b4")
    sel = elbow_df[elbow_df["K"] == k]
    ax.scatter(sel["K"], sel["Inertia"], s=180, color="red", zorder=3, label=f"Selected K = {k}")
    if suggested_k:
        ax.axvline(suggested_k, color="green", linestyle="--", label=f"Suggested K = {suggested_k}")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Inertia vs K")
    ax.set_xticks(list(K_RANGE))
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Look for the point where the curve starts to flatten (the 'elbow').")

with right:
    st.subheader(f"Customer Clusters (K = {k})")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("tab10")
    for cluster_id in sorted(clustered["Cluster"].unique()):
        pts = clustered[clustered["Cluster"] == cluster_id]
        ax.scatter(
            pts[FEATURES[0]], pts[FEATURES[1]],
            s=45, alpha=0.8, color=cmap(cluster_id), label=f"Cluster {cluster_id}",
        )
    if show_centroids:
        ax.scatter(
            centers[FEATURES[0]], centers[FEATURES[1]],
            s=250, marker="X", color="black", edgecolor="white", label="Centroids",
        )
    ax.set_xlabel(FEATURES[0])
    ax.set_ylabel(FEATURES[1])
    ax.set_title("Annual Income vs Spending Score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Silhouette score: **{sil_score:.3f}** (closer to 1 means better-separated clusters). "
        "Cluster IDs are arbitrary labels, not a ranking."
    )

# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------
st.header("3. Cluster Analysis")

profile = (
    clustered.groupby("Cluster")
    .agg(
        Customers=("CustomerID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=(FEATURES[0], "mean"),
        Avg_Spending=(FEATURES[1], "mean"),
    )
    .round(1)
    .rename(columns={
        "Avg_Age": "Mean Age",
        "Avg_Income": "Mean Income (k$)",
        "Avg_Spending": "Mean Spending Score",
    })
)

left, right = st.columns(2)
with left:
    st.subheader("Cluster Profiles")
    st.dataframe(profile)

with right:
    st.subheader("Gender Distribution per Cluster")
    gender_counts = pd.crosstab(clustered["Cluster"], clustered["Gender"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    gender_counts.plot(kind="bar", ax=ax, color=["#e377c2", "#1f77b4"], rot=0)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of customers")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Clustered data + download
# ---------------------------------------------------------------------------
st.header("4. Clustered Dataset")
selected_clusters = st.multiselect(
    "Filter by cluster", options=sorted(clustered["Cluster"].unique()),
    default=sorted(clustered["Cluster"].unique()),
)
st.dataframe(clustered[clustered["Cluster"].isin(selected_clusters)], hide_index=True)

st.download_button(
    "⬇️ Download clustered dataset (CSV)",
    data=clustered.to_csv(index=False).encode("utf-8"),
    file_name=f"mall_customers_k{k}.csv",
    mime="text/csv",
)
