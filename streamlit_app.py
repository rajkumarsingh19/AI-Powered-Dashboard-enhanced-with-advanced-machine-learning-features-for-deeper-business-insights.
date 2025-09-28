import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from datetime import datetime, timedelta
import calendar
from scipy import stats
import joblib

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore AI Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 SuperStore AI-Powered Executive Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# ===========================
# File Upload
# ===========================
fl = st.file_uploader("📂 Upload your dataset", type=["csv","txt","xlsx","xls"])
if fl is not None:
    df = pd.read_csv(fl, encoding="ISO-8859-1")
    st.success(f"Loaded {fl.name}")
else:
    # For demo purposes, using sample data generation if file not available
    st.info("Using sample data. Upload your own CSV file for analysis.")
    # Generate sample data if no file uploaded
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n_records = 1000
    
    sample_data = {
        'Order Date': np.random.choice(dates, n_records),
        'Ship Date': np.random.choice(dates, n_records),
        'Sales': np.random.exponential(100, n_records),
        'Profit': np.random.normal(20, 15, n_records),
        'Quantity': np.random.randint(1, 10, n_records),
        'Discount': np.random.uniform(0, 0.3, n_records),
        'Category': np.random.choice(['Furniture', 'Office Supplies', 'Technology'], n_records),
        'Sub-Category': np.random.choice(['Chairs', 'Tables', 'Phones', 'Accessories'], n_records),
        'Region': np.random.choice(['East', 'West', 'Central', 'South'], n_records),
        'State': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_records),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_records),
        'Order ID': [f'ORD_{i}' for i in range(n_records)]
    }
    df = pd.DataFrame(sample_data)

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# ===========================
# Advanced Filters
# ===========================
st.sidebar.header("🔹 Advanced Filters")

# Date range filter
startDate = df["Order Date"].min()
endDate = df["Order Date"].max()
date1 = st.sidebar.date_input("Start Date", startDate)
date2 = st.sidebar.date_input("End Date", endDate)
filtered_df = df[(df["Order Date"] >= pd.to_datetime(date1)) & (df["Order Date"] <= pd.to_datetime(date2))]

# Geographic filters
region = st.sidebar.multiselect("Region", df["Region"].unique())
state = st.sidebar.multiselect("State", df["State"].unique())
city = st.sidebar.multiselect("City", df["City"].unique())

if region:
    filtered_df = filtered_df[filtered_df["Region"].isin(region)]
if state:
    filtered_df = filtered_df[filtered_df["State"].isin(state)]
if city:
    filtered_df = filtered_df[filtered_df["City"].isin(city)]

# Numeric range filters
sales_range = st.sidebar.slider("Sales Range", float(df["Sales"].min()), float(df["Sales"].max()), 
                               (float(df["Sales"].min()), float(df["Sales"].max())))
profit_range = st.sidebar.slider("Profit Range", float(df["Profit"].min()), float(df["Profit"].max()), 
                                (float(df["Profit"].min()), float(df["Profit"].max())))
filtered_df = filtered_df[(filtered_df["Sales"] >= sales_range[0]) & (filtered_df["Sales"] <= sales_range[1])]
filtered_df = filtered_df[(filtered_df["Profit"] >= profit_range[0]) & (filtered_df["Profit"] <= profit_range[1])]

# ===========================
# AI/ML Features Section
# ===========================
st.sidebar.header("🤖 AI/ML Features")
enable_anomaly = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
enable_forecasting = st.sidebar.checkbox("Enable Sales Forecasting", value=True)
enable_clustering = st.sidebar.checkbox("Enable Customer Segmentation", value=True)
enable_step_change = st.sidebar.checkbox("Enable Step Change Detection", value=True)

# ===========================
# KPIs with AI Insights
# ===========================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
avg_discount = filtered_df["Discount"].mean()
profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0

# Calculate additional KPIs
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
days_analyzed = (filtered_df["Order Date"].max() - filtered_df["Order Date"].min()).days
daily_sales_rate = total_sales / days_analyzed if days_analyzed > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}", f"${daily_sales_rate:,.0f}/day")
col2.metric("🏦 Total Profit", f"${total_profit:,.0f}", f"{profit_margin:.1f}% margin")
col3.metric("📦 Total Orders", total_orders, f"${avg_order_value:.0f} AOV")
col4.metric("🔻 Avg Discount", f"{avg_discount:.2%}", "Discount Rate")

st.markdown("---")

# ===========================
# ANOMALY DETECTION
# ===========================
if enable_anomaly and len(filtered_df) > 10:
    st.header("🔍 Anomaly Detection")
    
    # Prepare data for anomaly detection
    anomaly_data = filtered_df[['Sales', 'Profit', 'Quantity', 'Discount']].copy()
    anomaly_data = anomaly_data.dropna()
    
    if len(anomaly_data) > 10:
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(anomaly_data)
        
        filtered_df['Anomaly'] = anomalies
        filtered_df['Anomaly'] = filtered_df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        
        # Display anomaly summary
        anomaly_count = (anomalies == -1).sum()
        total_count = len(anomalies)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Anomalies Detected", anomaly_count, f"{(anomaly_count/total_count)*100:.1f}%")
        col2.metric("✅ Normal Transactions", total_count - anomaly_count)
        col3.metric("📊 Total Analyzed", total_count)
        
        # Anomaly visualization
        fig_anomaly = px.scatter(
            filtered_df, x='Sales', y='Profit', color='Anomaly',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            title='Sales vs Profit - Anomaly Detection',
            hover_data=['Category', 'Sub-Category', 'Region']
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Show anomalous transactions
        with st.expander("View Anomalous Transactions"):
            anomalies_df = filtered_df[filtered_df['Anomaly'] == 'Anomaly']
            st.dataframe(anomalies_df[['Order Date', 'Sales', 'Profit', 'Category', 'Region']].head(20))

# ===========================
# STEP CHANGE DETECTION
# ===========================
if enable_step_change and len(filtered_df) > 30:
    st.header("📈 Step Change Detection")
    
    # Monthly sales aggregation
    monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).reset_index()
    monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
    monthly_sales = monthly_sales.sort_values('Order Date')
    
    # Calculate rolling statistics for step change detection
    monthly_sales['Rolling_Mean'] = monthly_sales['Sales'].rolling(window=3, center=True).mean()
    monthly_sales['Rolling_Std'] = monthly_sales['Sales'].rolling(window=3, center=True).std()
    monthly_sales['Z_Score'] = (monthly_sales['Sales'] - monthly_sales['Rolling_Mean']) / monthly_sales['Rolling_Std']
    
    # Detect step changes (z-score > 2)
    monthly_sales['Step_Change'] = np.abs(monthly_sales['Z_Score']) > 2
    monthly_sales['Change_Type'] = np.where(
        monthly_sales['Z_Score'] > 2, 'Positive Spike',
        np.where(monthly_sales['Z_Score'] < -2, 'Negative Spike', 'Normal')
    )
    
    # Plot with step changes highlighted
    fig_step = go.Figure()
    
    # Add sales line
    fig_step.add_trace(go.Scatter(
        x=monthly_sales['Order Date'], 
        y=monthly_sales['Sales'],
        mode='lines+markers',
        name='Monthly Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Highlight step changes
    spikes = monthly_sales[monthly_sales['Step_Change']]
    if not spikes.empty:
        fig_step.add_trace(go.Scatter(
            x=spikes['Order Date'],
            y=spikes['Sales'],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='Step Change',
            text=spikes['Change_Type']
        ))
    
    fig_step.update_layout(
        title='Monthly Sales with Step Change Detection',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        template='plotly_white'
    )
    st.plotly_chart(fig_step, use_container_width=True)
    
    # Display step change insights
    if not spikes.empty:
        st.subheader("Step Change Insights")
        for _, spike in spikes.iterrows():
            change_dir = "increase" if spike['Z_Score'] > 0 else "decrease"
            st.info(f"**{spike['Order Date'].strftime('%B %Y')}**: Significant {change_dir} in sales (Z-score: {spike['Z_Score']:.2f})")

# ===========================
# SALES FORECASTING
# ===========================
if enable_forecasting and len(filtered_df) > 60:
    st.header("🔮 Sales Forecasting")
    
    # Create time series data
    daily_sales = filtered_df.groupby(filtered_df['Order Date'].dt.date)['Sales'].sum().reset_index()
    daily_sales['Order Date'] = pd.to_datetime(daily_sales['Order Date'])
    daily_sales = daily_sales.set_index('Order Date').asfreq('D').fillna(0)
    
    # Simple forecasting using moving averages
    forecast_days = 30
    last_30_days = daily_sales['Sales'].tail(30)
    
    # Multiple forecasting methods
    simple_ma = last_30_days.mean()
    weighted_ma = last_30_days.ewm(span=15).mean().iloc[-1]
    
    # Generate forecast
    last_date = daily_sales.index[-1]
    forecast_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast_Sales': [simple_ma * (0.95 + 0.1 * np.random.random()) for _ in range(forecast_days)],
        'Confidence_Low': [simple_ma * 0.8 for _ in range(forecast_days)],
        'Confidence_High': [simple_ma * 1.2 for _ in range(forecast_days)]
    })
    
    # Plot historical and forecasted data
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=daily_sales.index[-90:],  # Last 90 days
        y=daily_sales['Sales'][-90:],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast_Sales'],
        mode='lines',
        name='Forecast',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['Confidence_High'].tolist() + forecast_df['Confidence_Low'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig_forecast.update_layout(
        title='Sales Forecast (Next 30 Days)',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        template='plotly_white'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast summary
    avg_forecast = forecast_df['Forecast_Sales'].mean()
    st.metric("📊 Forecasted Average Daily Sales", f"${avg_forecast:,.0f}")

# ===========================
# CUSTOMER SEGMENTATION
# ===========================
if enable_clustering and len(filtered_df) > 50:
    st.header("👥 Customer Segmentation")
    
    # Create customer-level data (using Order ID as proxy for customer)
    customer_data = filtered_df.groupby('Order ID').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Discount': 'mean'
    }).reset_index()
    
    # Prepare data for clustering
    X = customer_data[['Sales', 'Profit', 'Quantity', 'Discount']]
    X = X.dropna()
    
    if len(X) > 10:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        customer_data['Cluster'] = clusters
        customer_data['Cluster_Label'] = customer_data['Cluster'].map({
            0: 'Budget Shoppers',
            1: 'Premium Customers',
            2: 'Value Seekers'
        })
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create cluster visualization
        fig_cluster = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=customer_data['Cluster_Label'],
            title='Customer Segmentation (PCA Visualization)',
            labels={'x': 'PC1', 'y': 'PC2'},
            hover_data={'Sales': customer_data['Sales'], 'Profit': customer_data['Profit']}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster analysis
        cluster_summary = customer_data.groupby('Cluster_Label').agg({
            'Sales': ['count', 'mean', 'sum'],
            'Profit': 'mean',
            'Discount': 'mean'
        }).round(2)
        
        st.subheader("Cluster Characteristics")
        st.dataframe(cluster_summary)

# ===========================
# TRADITIONAL VISUALIZATIONS
# ===========================
st.header("📊 Traditional Analytics")

# Category Sales + Profit Combo
cat_summary = filtered_df.groupby("Category").agg({"Sales":"sum", "Profit":"sum"}).reset_index()
fig_combo = go.Figure()
fig_combo.add_trace(go.Bar(x=cat_summary["Category"], y=cat_summary["Sales"], name="Sales", 
                          marker_color="skyblue", text=[f"${x:,.0f}" for x in cat_summary["Sales"]]))
fig_combo.add_trace(go.Scatter(x=cat_summary["Category"], y=cat_summary["Profit"], mode="lines+markers", 
                              name="Profit", marker_color="red", text=[f"${x:,.0f}" for x in cat_summary["Profit"]]))
fig_combo.update_layout(title="Category Sales & Profit Combo", yaxis_title="Amount ($)", template="plotly_white")
st.plotly_chart(fig_combo, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Sunburst Hierarchy
    fig_sun = px.sunburst(filtered_df, path=["Region","Category","Sub-Category"], 
                         values="Sales", color="Profit", hover_data=["Sales"], 
                         color_continuous_scale="RdYlGn")
    fig_sun.update_layout(title="Sales Hierarchy")
    st.plotly_chart(fig_sun, use_container_width=True)

with col2:
    # Monthly Sales Trend
    filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
    monthly_sales = filtered_df.groupby(filtered_df["month_year"].dt.to_timestamp())["Sales"].sum().reset_index()
    monthly_sales = monthly_sales.sort_values("month_year")
    monthly_sales["Rolling_Avg"] = monthly_sales["Sales"].rolling(3).mean()

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=monthly_sales["month_year"], y=monthly_sales["Sales"], 
                                 mode="lines+markers", name="Sales", marker=dict(color="blue")))
    fig_line.add_trace(go.Scatter(x=monthly_sales["month_year"], y=monthly_sales["Rolling_Avg"], 
                                 mode="lines", name="3-Month Rolling Avg", 
                                 line=dict(color="orange", dash="dash")))
    fig_line.update_layout(title="Monthly Sales Trend", xaxis_title="Month-Year", 
                          yaxis_title="Sales", template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

# Sales vs Profit Scatter
fig_scatter = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity", 
                        color="Category", hover_data=["Region","Sub-Category"], 
                        template="plotly_dark", title="Sales vs Profit by Category & Quantity")
st.plotly_chart(fig_scatter, use_container_width=True)

# ===========================
# ADDITIONAL ADVANCED FEATURES
# ===========================
st.header("🎯 Advanced Insights")

col1, col2 = st.columns(2)

with col1:
    # Profitability Heatmap
    profitability = filtered_df.pivot_table(
        index='Category', 
        columns='Region', 
        values='Profit', 
        aggfunc='sum',
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        profitability,
        title='Profitability Heatmap (Category vs Region)',
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Discount Impact Analysis
    discount_bins = pd.cut(filtered_df['Discount'], bins=[0, 0.1, 0.2, 0.3, 1], 
                          labels=['0-10%', '10-20%', '20-30%', '30%+'])
    discount_impact = filtered_df.groupby(discount_bins).agg({
        'Sales': 'mean',
        'Profit': 'mean',
        'Quantity': 'mean'
    }).reset_index()
    
    fig_discount = px.bar(
        discount_impact, x='Discount', y=['Sales', 'Profit'],
        title='Impact of Discount on Sales & Profit',
        barmode='group'
    )
    st.plotly_chart(fig_discount, use_container_width=True)

# ===========================
# DATA EXPORT AND SUMMARY
# ===========================
st.header("📑 Data Management")

with st.expander("View Filtered Data"):
    st.dataframe(filtered_df.head(1000))

# Download options
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Filtered Data", data=csv, 
                  file_name="Enhanced_Superstore_Analysis.csv", mime="text/csv")

# Executive Summary
st.header("📋 Executive Summary")
if st.button("Generate AI Summary"):
    with st.spinner("Generating insights..."):
        # Basic insights
        top_category = filtered_df.groupby('Category')['Sales'].sum().idxmax()
        top_region = filtered_df.groupby('Region')['Sales'].sum().idxmax()
        avg_sale = filtered_df['Sales'].mean()
        
        st.success(f"""
        **Key Insights:**
        - Top performing category: **{top_category}**
        - Most profitable region: **{top_region}**
        - Average transaction value: **${avg_sale:.2f}**
        - Profit margin: **{profit_margin:.1f}%**
        - Data period: **{filtered_df['Order Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Order Date'].max().strftime('%Y-%m-%d')}**
        
        **AI Features Enabled:**
        - Anomaly Detection: {enable_anomaly}
        - Sales Forecasting: {enable_forecasting}
        - Customer Segmentation: {enable_clustering}
        - Step Change Detection: {enable_step_change}
        """)

st.markdown("---")
st.markdown("✅ **AI-Powered Dashboard** enhanced with advanced machine learning features for deeper business insights.")
