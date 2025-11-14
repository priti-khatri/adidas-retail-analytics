import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from kaggle.api.kaggle_api_extended import KaggleApi

# config
st.set_page_config(
    page_title="Adidas USA Retail Analytics",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# data load from kaggle dataset
@st.cache_data
def load_kaggle_data():
    # Inject Kaggle API keys from Streamlit secrets
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

    api = KaggleApi()
    api.authenticate()

    dataset = "thedevastator/adidas-fashion-retail-products-dataset-9300-prod"

    api.dataset_download_files(dataset, path="data/", unzip=True)

    df = pd.read_csv("data/adidas_usa.csv")
    df['Date'] = pd.to_datetime(df['date'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['total_revenue'] = df['price'] * df['quantity']
    df.dropna(subset=["Date", "price", "quantity"], inplace=True)
    return df

df = load_kaggle_data()


# filters
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/2/20/Adidas_Logo.svg", 
    width=150
)
st.sidebar.title("🎛️ Filters")

category = st.sidebar.multiselect(
    "Select Category", df['category'].unique(), default=df['category'].unique()
)
gender = st.sidebar.multiselect(
    "Select Gender", df['gender'].unique(), default=df['gender'].unique()
)
brand = st.sidebar.multiselect(
    "Select Brand", df['brand'].unique(), default=df['brand'].unique()
)

df_filtered = df[
    (df['category'].isin(category)) &
    (df['gender'].isin(gender)) &
    (df['brand'].isin(brand))
]


# anomalies detected
def detect_anomalies(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    df_anom = data.copy()
    df_anom['anomaly_score'] = model.fit_predict(df_anom[['total_revenue']])
    df_anom['anomaly'] = df_anom['anomaly_score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return df_anom

df_anomaly = detect_anomalies(df_filtered)


# kpi
total_revenue = df_filtered['total_revenue'].sum() / 1e3
avg_price = df_filtered['price'].mean()
total_anomalies = df_anomaly[df_anomaly['anomaly'] == "Anomaly"].shape[0]
unique_customers = df_filtered['sku'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue (K USD)", f"${total_revenue:.2f}K")
col2.metric("📦 Avg Product Price", f"${avg_price:.2f}")
col3.metric("⚠️ Revenue Anomalies", total_anomalies)
col4.metric("🧩 Unique SKUs", unique_customers)


# ml forecast

#  A) DEMAND FORECASTING (RandomForest ML—not rolling mean)
def forecast_demand(df):
    df_daily = df.groupby("Date")["quantity"].sum().reset_index()
    df_daily["day_num"] = df_daily["Date"].map(lambda x: x.toordinal())

    X = df_daily[["day_num"]]
    y = df_daily["quantity"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    future_dates = pd.date_range(df_daily["Date"].max(), periods=30)
    future_df = pd.DataFrame({"Date": future_dates})
    future_df["day_num"] = future_df["Date"].map(lambda x: x.toordinal())
    future_df["forecast_qty"] = model.predict(future_df[["day_num"]])

    return df_daily, future_df

df_daily, df_future = forecast_demand(df_filtered)

#  B) PROMOTION EFFECT MODEL (Regression)
def promo_effect_model(df):
    # Fake 'promotion flag' if not present (you can replace this with real promo field)
    df["promotion"] = (df["price"] < df["price"].median()).astype(int)

    X = df[["promotion"]]
    y = df["total_revenue"]

    model = RandomForestRegressor()
    model.fit(X, y)

    promo_lift = model.feature_importances_[0]
    return promo_lift

promo_lift_score = promo_effect_model(df_filtered)

# C) PRODUCT RECOMMENDER (simple popularity-based scoring)
def recommend_products(df, top_n=5):
    scores = (
        df.groupby("sku")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return scores

top_products = recommend_products(df_filtered)


# tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Revenue & Anomalies", "Sales Trends", "Customer Segmentation", "Forecasting", "Promo & Recommendations"]
)

with tab1:
    fig_rev = px.line(
        df_anomaly, x="Date", y="total_revenue", color="category",
        title="Revenue Trend with Anomalies"
    )
    fig_rev.add_scatter(
        x=df_anomaly[df_anomaly["anomaly"] == "Anomaly"]["Date"],
        y=df_anomaly[df_anomaly["anomaly"] == "Anomaly"]["total_revenue"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Anomaly"
    )
    st.plotly_chart(fig_rev, use_container_width=True)

with tab2:
    sales_chart = px.bar(
        df_filtered.groupby(['gender','category'])['total_revenue'].sum().reset_index(),
        x='category', y='total_revenue', color='gender', barmode='group',
        title="Revenue by Gender and Category"
    )
    st.plotly_chart(sales_chart, use_container_width=True)

with tab3:
    cust_data = df_filtered.groupby('sku')[['total_revenue', 'quantity']].sum().reset_index()
    scaler = StandardScaler()
    cust_scaled = scaler.fit_transform(cust_data[['total_revenue', 'quantity']])
    kmeans = KMeans(n_clusters=3, random_state=42).fit(cust_scaled)
    cust_data['Segment'] = kmeans.labels_

    seg_chart = px.scatter(
        cust_data, x='total_revenue', y='quantity', color='Segment',
        title="SKU Segmentation (Revenue vs Quantity)"
    )
    st.plotly_chart(seg_chart, use_container_width=True)

with tab4:
    fig_fc = px.line(df_daily, x="Date", y="quantity", title="Historical Demand")
    fig_fc2 = px.line(df_future, x="Date", y="forecast_qty", title="30-Day Demand Forecast")

    st.plotly_chart(fig_fc, use_container_width=True)
    st.plotly_chart(fig_fc2, use_container_width=True)

with tab5:
    st.subheader("📈 Promotion Effect Score")
    st.write(f"**Predicted Promo Lift: {promo_lift_score:.3f}** (higher = stronger impact)")

    st.subheader("🏆 Top Recommended Products (High Revenue SKUs)")
    st.write(top_products)


# footer added
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;'>Built with ❤️ using Python, Streamlit, Plotly & ML — Adidas Analytics</div>",
    unsafe_allow_html=True
)
