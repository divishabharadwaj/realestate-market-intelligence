import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set elegant dark page schema
st.set_page_config(page_title="Parcl Buyer Profiler", layout="wide")
st.title("🏡 Real Estate Buyer Intelligence & ML Segmentation")

# Loader logic
@st.cache_data
def load_data():
    # Simulated mapping of PDF OCR records & property transactions
    buyers = pd.DataFrame({
        'client_id': [f"C{i:04d}" for i in range(1, 51)],
        'client_type': ['Individual' if i % 4 != 0 else 'Company' for i in range(1, 51)],
        'age': np.random.randint(25, 80, 50),
        'satisfaction': np.random.randint(1, 6, 50),
        'country': np.random.choice(['USA', 'Canada', 'Germany', 'Belgium'], 50),
        'region': np.random.choice(['California', 'Nevada', 'Utah', 'Colorado'], 50),
        'purpose': np.random.choice(['Home', 'Investment', 'Personal use'], 50),
        'loan_applied': np.random.choice(['Yes', 'No'], 50),
        'property_value': np.random.randint(150000, 750000, 50)
    })
    return buyers

df = load_data()

# Sidebar Controllers
st.sidebar.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?q=80&w=200", caption="Parcl Research Co.")
st.sidebar.header("🎯 Data Filters")
f_country = st.sidebar.selectbox("Country of Origin", ["All"] + list(df['country'].unique()))
f_purpose = st.sidebar.selectbox("Investment Purpose", ["All"] + list(df['purpose'].unique()))

filtered_df = df.copy()
if f_country != "All":
    filtered_df = filtered_df[filtered_df['country'] == f_country]
if f_purpose != "All":
    filtered_df = filtered_df[filtered_df['purpose'] == f_purpose]

# ML Clustering block
st.header("🤖 Automatic ML Buyer Profiler (K-Means)")
k_clusters = st.slider("Select Target Segment Count (K)", 2, 8, 4)

features = filtered_df[['age', 'satisfaction', 'property_value']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=k_clusters, random_state=42)
filtered_df['cluster'] = kmeans.fit_predict(scaled_features)
filtered_df['cluster'] = filtered_df['cluster'].apply(lambda x: f"Segment {x+1}")

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(
        filtered_df, x="age", y="property_value", color="cluster",
        size="satisfaction", hover_data=['client_id', 'country', 'purpose'],
        title="Buyer Segments: Age vs Property Investment Value",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💡 Discovered Segment Intelligence")
    segment_stats = filtered_df.groupby('cluster').agg({
        'client_id': 'count',
        'age': 'mean',
        'property_value': 'mean',
        'satisfaction': 'mean'
    }).rename(columns={'client_id': 'Total Buyers', 'age': 'Mean Age', 'property_value': 'Avg Value', 'satisfaction': 'Satisfaction'})
    st.dataframe(segment_stats.style.format({'Mean Age': '{:.1f} yr', 'Avg Value': '${:,.2f}', 'Satisfaction': '{:.1f} / 5'}))

# Country Map representation
st.header("🌎 Territory Coverage Overview")
map_fig = px.histogram(filtered_df, x="region", color="cluster", barmode="group", title="Buyers mapped to Region territories")
st.plotly_chart(map_fig, use_container_width=True)

