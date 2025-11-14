import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import kaggle
import os

# css for mordern design
st.markdown("""
    <style>
    
    /* GENERAL STYLES */
    body {
        background-color: #121212; /* Dark background for the body */
        color: white;  /* White text color */
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* --- Sidebar Design (Glassmorphism) --- */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1); /* Glass effect */
        backdrop-filter: blur(10px); /* Background blur effect */
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Sidebar title and image */
    .css-1r2zvgr {
        color: white !important;
    }

    .stSidebar > div:first-child {
        background: transparent !important;
    }

    /* --- Filters (Selectbox, Multiselect) --- */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.2) !important;  /* Transparent black */
        color: white !important; /* White text */
        border: 2px solid #0c5a94 !important; /* Neon blue border */
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(12, 90, 148, 0.8); /* Neon blue glow */
    }

    /* Selected value text color */
    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* Dropdown background and hover */
    ul[role="listbox"] {
        background-color: rgba(0, 0, 0, 0.6) !important; /* Darker dropdown */
        color: white !important;
    }

    li[role="option"]:hover {
        background-color: #0c5a94 !important;  /* Neon blue hover */
        color: white !important;
    }

    /* Arrow icon color */
    div[data-baseweb="select"] svg {
        fill: white !important;
    }

    /* --- Sidebar Filter Button Styles --- */
    .stButton > button {
        background-color: #0c5a94 !important;  /* Neon blue background */
        color: white !important;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(12, 90, 148, 0.8); /* Neon blue glow */
        border: 2px solid #0c5a94;
        padding: 0.5rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #073c56 !important;  /* Darker blue for hover */
        box-shadow: 0 0 15px rgba(12, 90, 148, 0.8);
    }

    /* --- Title Styling --- */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: bold;
        color: #ffffff !important;
    }

    /* --- Page content background --- */
    .stApp {
        background: #121212 !important;
    }

    </style>
""", unsafe_allow_html=True)

# config
st.set_page_config(page_title="Adidas Product Intelligence", page_icon="👟", layout="wide")

# data load from kaggle dataset
@st.cache_data
def load_kaggle_data():
    # set kaggle API credentials from secrets
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

    dataset_name = "thedevastator/adidas-fashion-retail-products-dataset-9300-prod"
    folder = "data/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # download dataset if file not already present
    if not os.path.exists("data/adidas_usa.csv"):
        kaggle.api.dataset_download_files(dataset_name, path=folder, unzip=True)

    df = pd.read_csv("data/adidas_usa.csv")

    # clean required fields
    df.dropna(subset=["selling_price", "name", "category", "brand"], inplace=True)
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")

    return df[df["selling_price"].notna()]


df = load_kaggle_data()


# filters
st.sidebar.title("🔎 Filters")

categories = st.sidebar.multiselect("Category", df["category"].unique(), default=df["category"].unique())
brands = st.sidebar.multiselect("Brand", df["brand"].unique(), default=df["brand"].unique())
colors = st.sidebar.multiselect("Color", df["color"].dropna().unique(), default=df["color"].dropna().unique())

df_filtered = df[
    df["category"].isin(categories) &
    df["brand"].isin(brands) &
    df["color"].isin(colors)
]

# title
st.title("👟 Adidas Product Intelligence Dashboard")
st.caption("Automated dataset from Kaggle + Segmentation + Anomaly Detection + Recommendations")

# kpi
col1, col2, col3 = st.columns(3)

col1.metric("Total Products", df_filtered.shape[0])
col2.metric("Average Price", f"${df_filtered['selling_price'].mean():.2f}")
col3.metric("Avg Rating", f"{df_filtered['average_rating'].mean():.2f}")

# tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Price Analysis",
    "Category Analysis",
    "Segmentation",
    "Anomaly Detection",
    "Recommendations",
])

# price pointers
with tab1:
    st.subheader("💰 Price Distribution")
    fig_price = px.histogram(df_filtered, x="selling_price", nbins=40, template="plotly_white")
    st.plotly_chart(fig_price, use_container_width=True)

# categories
with tab2:
    st.subheader("📦 Product Categories")
    category_counts = df_filtered["category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]

    fig_cat = px.bar(category_counts, x="Category", y="Count", template="plotly_white")
    st.plotly_chart(fig_cat, use_container_width=True)

# segmentation
with tab3:
    st.subheader("🔍 Product Segmentation (KMeans)")

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_filtered[["selling_price", "average_rating"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered["Segment"] = kmeans.fit_predict(df_scaled)

    fig_seg = px.scatter(
        df_filtered,
        x="selling_price",
        y="average_rating",
        color="Segment",
        hover_data=["name", "brand"],
        title="Product Segmentation by Price & Rating",
        template="plotly_white"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

# anomalies
with tab4:
    st.subheader("⚠️ Anomalous Products (Isolation Forest)")

    iso = IsolationForest(contamination=0.05, random_state=42)
    df_filtered["Anomaly"] = iso.fit_predict(df_filtered[["selling_price", "average_rating"]])

    anomalies = df_filtered[df_filtered["Anomaly"] == -1]

    fig_anom = px.scatter(
        anomalies,
        x="selling_price",
        y="average_rating",
        hover_data=["name", "brand"],
        color="Anomaly",
        title="Detected Price/Rating Anomalies",
        template="plotly_white"
    )
    st.plotly_chart(fig_anom, use_container_width=True)

# recommendations
with tab5:
    st.subheader("🏷️ Product Recommendations")

    product_list = df_filtered["name"].tolist()
    selected_product = st.selectbox("Select a product:", product_list)

    base = df_filtered[df_filtered["name"] == selected_product].iloc[0]

    recommended = df_filtered[
        (df_filtered["category"] == base["category"]) &
        (df_filtered["selling_price"].between(base["selling_price"] - 20, base["selling_price"] + 20)) &
        (df_filtered["name"] != selected_product)
    ].sort_values("average_rating", ascending=False).head(5)

    st.write("### Recommended Products")
    st.write(recommended[["name", "selling_price", "average_rating", "brand"]])

    # image gallery
    if "images" in recommended.columns:
        for _, row in recommended.iterrows():
            imgs = str(row["images"]).split("~")
            st.image(imgs[0], width=250, caption=row["name"])


# footer added
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;'>Built with ❤️ using Python, Streamlit, Plotly & ML — Adidas Analytics</div>",
    unsafe_allow_html=True
)
