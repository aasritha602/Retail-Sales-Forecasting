import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1,h2,h3{
    color:#1F4E79;
}

div[data-testid="metric-container"]{
    background-color:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #E6E6E6;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""",unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():

    sales = pd.read_csv("train.csv")

    sales["Order Date"] = pd.to_datetime(
        sales["Order Date"],
        dayfirst=True
    )

    sales["Ship Date"] = pd.to_datetime(
        sales["Ship Date"],
        dayfirst=True
    )

    sales["Year"] = sales["Order Date"].dt.year
    sales["Month"] = sales["Order Date"].dt.month
    sales["Month Name"] = sales["Order Date"].dt.month_name()

    forecast = pd.read_csv("forecast.csv")

    monthly = pd.read_csv("monthly_sales.csv")

    monthly["Order Date"] = pd.to_datetime(
        monthly["Order Date"]
    )

    anomalies = pd.read_csv("anomalies.csv")

    anomalies["Order Date"] = pd.to_datetime(
        anomalies["Order Date"]
    )

    comparison = pd.read_csv("comparison.csv")

    comparison["Order Date"] = pd.to_datetime(
        comparison["Order Date"]
    )

    clusters = pd.read_csv("clusters.csv")

    return (
        sales,
        forecast,
        monthly,
        anomalies,
        comparison,
        clusters
    )

sales,forecast,monthly,anomalies,comparison,clusters = load_data()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Dashboard",

    [

        "🏠 Home",

        "📈 Forecast Explorer",

        "🚨 Anomaly Detection",

        "📦 Product Segmentation"

    ]

)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------

if page=="🏠 Home":

    st.title("📊 Retail Sales Forecasting Dashboard")

    st.write(
        """
        This dashboard presents an intelligent retail sales forecasting system
        developed using Time Series Forecasting, Machine Learning,
        Anomaly Detection and Product Demand Segmentation.
        """
    )

    # ---------------- KPI ----------------

    total_sales = sales["Sales"].sum()

    total_orders = len(sales)

    avg_sales = sales["Sales"].mean()

    total_categories = sales["Category"].nunique()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "🛒 Orders",
        total_orders
    )

    c3.metric(
        "📦 Categories",
        total_categories
    )

    c4.metric(
        "💵 Avg Order Value",
        f"${avg_sales:.2f}"
    )

    st.divider()

    # ---------------- SALES BY YEAR ----------------

    st.subheader("Total Sales by Year")

    yearly = (

        sales.groupby("Year")["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        yearly,

        x="Year",

        y="Sales",

        color="Sales",

        text_auto=".2s",

        title="Yearly Sales"

    )

    fig.update_layout(

        xaxis_title="Year",

        yaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------- MONTHLY TREND ----------------

    st.subheader("Monthly Sales Trend")

    fig = px.line(

        monthly,

        x="Order Date",

        y="Sales",

        markers=True,

        title="Monthly Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------- CATEGORY SALES ----------------

    col1,col2 = st.columns(2)

    with col1:

        category_sales = (

            sales.groupby("Category")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.pie(

            category_sales,

            names="Category",

            values="Sales",

            hole=.45,

            title="Sales by Category"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        region_sales = (

            sales.groupby("Region")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            region_sales,

            x="Region",

            y="Sales",

            color="Region",

            title="Sales by Region"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(sales.head(15),use_container_width=True)
# ==========================================================
# FORECAST EXPLORER
# ==========================================================

elif page == "📈 Forecast Explorer":

    st.title("📈 Sales Forecast Explorer")

    st.markdown("""
    This page presents the sales forecasting results generated using the
    **XGBoost Regression Model**, which achieved the best forecasting
    accuracy during model comparison.
    """)

    st.divider()

    # ---------------- KPI ----------------

    total_actual = forecast["Actual"].sum()
    total_pred = forecast["Predicted"].sum()

    avg_actual = forecast["Actual"].mean()
    avg_pred = forecast["Predicted"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Actual Sales",
        f"${total_actual:,.0f}"
    )

    c2.metric(
        "Predicted Sales",
        f"${total_pred:,.0f}"
    )

    c3.metric(
        "Average Actual",
        f"${avg_actual:,.2f}"
    )

    c4.metric(
        "Average Prediction",
        f"${avg_pred:,.2f}"
    )

    st.divider()

    # ------------------------------------------------

    st.subheader("Actual vs Predicted Sales")

    plot_df = forecast.copy()

    plot_df["Month"] = range(1, len(plot_df)+1)

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=plot_df["Month"],

            y=plot_df["Actual"],

            mode="lines+markers",

            name="Actual"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=plot_df["Month"],

            y=plot_df["Predicted"],

            mode="lines+markers",

            name="Predicted"

        )

    )

    fig.update_layout(

        title="Actual vs Predicted Sales",

        xaxis_title="Forecast Period",

        yaxis_title="Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------

    st.subheader("Forecast Comparison Table")

    st.dataframe(
        forecast,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------

    st.subheader("Forecast Accuracy")

    error = abs(
        forecast["Actual"] -
        forecast["Predicted"]
    )

    forecast["Absolute Error"] = error

    fig = px.bar(

        forecast,

        y="Absolute Error",

        x=forecast.index,

        color="Absolute Error",

        title="Prediction Error"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------

    st.subheader("Forecast Distribution")

    fig = px.histogram(

        forecast,

        x="Predicted",

        nbins=20,

        title="Distribution of Forecasted Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------

    st.subheader("Business Interpretation")

    st.success("""
The forecasting model predicts future sales with a relatively small
difference between actual and predicted values. The trend indicates
that the model captures the overall sales behaviour effectively,
making it suitable for inventory planning, demand estimation,
and procurement decisions.
""")

    st.divider()

    # ------------------------------------------------

    st.subheader("Model Performance")

    st.info("""
✔ Best Model : XGBoost Regressor

✔ Forecast Horizon : 3 Months

✔ Evaluation Metrics Used

• MAE

• RMSE

• MAPE

The XGBoost model achieved the lowest forecasting error among all
evaluated models and is therefore recommended for production use.
""")

    st.divider()

    # ------------------------------------------------

    st.download_button(

        label="⬇ Download Forecast Results",

        data=forecast.to_csv(index=False),

        file_name="forecast_results.csv",

        mime="text/csv"

    )
# ==========================================================
# ANOMALY DETECTION
# ==========================================================

elif page == "🚨 Anomaly Detection":

    st.title("🚨 Sales Anomaly Detection")

    st.write("""
    Isolation Forest and Z-Score techniques were used to detect
    unusually high or low weekly sales.
    """)

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Records",
        len(anomalies)
    )

    c2.metric(
        "Detected Anomalies",
        len(anomalies[anomalies["Anomaly_IF"] == -1])
    )

    st.divider()

    st.subheader("Weekly Sales")

    fig = px.line(
        anomalies,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Weekly Sales Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Isolation Forest Results")

    colors = anomalies["Anomaly_IF"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    fig = px.scatter(
        anomalies,
        x="Order Date",
        y="Sales",
        color=colors,
        title="Isolation Forest Detection"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Z-Score Distribution")

    fig = px.bar(
        anomalies,
        x="Order Date",
        y="ZScore",
        color="ZScore"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Detected Records")

    st.dataframe(
        anomalies,
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Anomaly Report",
        anomalies.to_csv(index=False),
        "anomalies.csv",
        "text/csv"
    )
# ==========================================================
# PRODUCT DEMAND SEGMENTATION
# ==========================================================

elif page == "📦 Product Segmentation":

    st.title("📦 Product Demand Segmentation")

    st.write("""
    Product sub-categories were grouped using K-Means Clustering
    based on Total Sales, Average Sales, Sales Variability,
    and Order Count.
    """)

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Sub-Categories",
        len(clusters)
    )

    c2.metric(
        "Clusters",
        clusters["Cluster"].nunique()
    )

    c3.metric(
        "Highest Sales",
        f"${clusters['Total_Sales'].max():,.0f}"
    )

    st.divider()

    st.subheader("Cluster Distribution")

    fig = px.scatter(
        clusters,
        x="Average_Sales",
        y="Total_Sales",
        color=clusters["Cluster"].astype(str),
        size="Order_Count",
        hover_name="Sub-Category",
        title="Product Demand Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Products by Cluster")

    st.dataframe(
        clusters.sort_values("Cluster"),
        use_container_width=True
    )

    st.divider()

    st.subheader("Cluster-wise Total Sales")

    cluster_sales = (
        clusters
        .groupby("Cluster")["Total_Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        cluster_sales,
        x="Cluster",
        y="Total_Sales",
        color="Cluster",
        title="Sales Contribution by Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Top Performing Products")

    top_products = (
        clusters
        .sort_values(
            by="Total_Sales",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_products,
        x="Sub-Category",
        y="Total_Sales",
        color="Cluster",
        title="Top 10 Product Sub-Categories"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.success("""
### Business Interpretation

• High-sales clusters should receive priority inventory allocation.

• Medium-demand clusters require regular replenishment.

• Low-demand products should be stocked conservatively to reduce holding costs.

• Cluster-based inventory planning can improve warehouse efficiency and reduce stock shortages.
""")

    st.download_button(
        "⬇ Download Cluster Report",
        clusters.to_csv(index=False),
        file_name="clusters.csv",
        mime="text/csv"
    )
# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
### 📄 Project Information

**Project Title:** Intelligent Retail Sales Forecasting & Inventory Planning

**Forecasting Models Used**
- SARIMA
- Prophet
- XGBoost

**Analytics Techniques**
- Time Series Analysis
- Isolation Forest
- Z-Score Anomaly Detection
- K-Means Clustering
- PCA Visualization

**Developed By:** Aasritha (Honey)

**Internship Project**
"""
)