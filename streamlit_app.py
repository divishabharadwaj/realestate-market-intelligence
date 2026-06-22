import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set elegant dark page schema
st.set_page_config(page_title="Real Estate Market Intelligence", layout="wide", initial_sidebar_state="expanded")
st.title("🏡 Real Estate Buyer Intelligence & ML Segmentation")

# Loader logic generating 2,000 clients and 10,000 properties as required
@st.cache_data
def load_data():
    np.random.seed(42)
    
    # 1. Generate 2,000 Clients (Buyers)
    countries_regions = {
        'USA': ['California', 'Virginia', 'Utah', 'Colorado', 'Nevada', 'New York', 'Washington', 'Florida', 'Ohio', 'Oregon'],
        'Canada': ['Quebec', 'British Columbia', 'Ontario', 'Manitoba', 'Alberta'],
        'Germany': ['Berlin', 'Hamburg', 'North Rhine', 'Bavaria'],
        'Belgium': ['Brussels', 'Antwerp', 'Flanders'],
        'UK': ['England', 'Scotland', 'Wales'],
        'Australia': ['Victoria', 'New South Wales', 'Queensland']
    }
    
    first_names_m = ['Grant', 'Franklin', 'Arthur', 'Cole', 'Conner', 'Xavier', 'Jace', 'Parker', 'Charles', 'Paul', 'Justin', 'Gianni', 'Walter', 'Peter', 'Nicholas', 'David', 'Stephen', 'Brandon', 'Samuel', 'Kenneth', 'Zachary', 'Michael', 'Lawrence']
    first_names_f = ['Madalyn', 'Hazel', 'Janelle', 'Laurel', 'Judy', 'Amanda', 'Deborah', 'Danielle', 'Sarah', 'Dorothy', 'Ann', 'Patricia', 'Cynthia', 'Katherine', 'Ruth', 'Helen']
    last_names = ['Weber', 'Mack', 'Bray', 'Mercer', 'Ayers', 'Taylor', 'Huff', 'Faulkner', 'Owen', 'Espinoza', 'Benitez', 'Riggs', 'Poole', 'Allen', 'Turner', 'Griffin', 'Peterson', 'Fritz', 'Adams', 'Williams', 'Jordan', 'Davis', 'Stone', 'Alexander', 'Hansen', 'Brown', 'Bennett', 'Pierce', 'Cooper', 'Jones', 'Thomas', 'Howard', 'Cox', 'Jackson', 'Martin', 'Bryant', 'Young', 'Scott']
    
    client_ids = [f"C{i:04d}" for i in range(1, 2001)]
    client_types = []
    genders = []
    fnames = []
    lnames = []
    countries = []
    regions = []
    purposes = []
    loans = []
    channels = []
    ages = []
    satisfactions = []
    
    for i in range(1, 2001):
        ct = 'Corporate' if i % 15 == 0 else ('Company' if i % 8 == 0 else 'Individual')
        gender = np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04]) if ct == 'Individual' else 'Other'
        fn = np.random.choice(first_names_m) if gender == 'M' else (np.random.choice(first_names_f) if gender == 'F' else 'N/A')
        ln = np.random.choice(last_names) if ct == 'Individual' else np.random.choice(['Group', 'Co', 'Holdings', 'Ventures', 'Inc', 'Ltd'])
        
        country = np.random.choice(list(countries_regions.keys()), p=[0.65, 0.15, 0.08, 0.04, 0.05, 0.03])
        region = np.random.choice(countries_regions[country])
        purpose = np.random.choice(['Home', 'Investment', 'Personal use'], p=[0.55, 0.35, 0.10])
        loan = np.random.choice(['Yes', 'No'], p=[0.45, 0.55])
        channel = np.random.choice(['Website', 'Agency', 'Client', 'Direct', 'Social Media'], p=[0.35, 0.25, 0.15, 0.15, 0.10])
        age = int(np.random.randint(25, 80)) if ct == 'Individual' else int(np.random.randint(5, 30))
        satisfaction = int(np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.15, 0.25, 0.30, 0.20]))
        
        client_types.append(ct)
        genders.append(gender)
        fnames.append(fn)
        lnames.append(ln)
        countries.append(country)
        regions.append(region)
        purposes.append(purpose)
        loans.append(loan)
        channels.append(channel)
        ages.append(age)
        satisfactions.append(satisfaction)
        
    buyers = pd.DataFrame({
        'client_id': client_ids,
        'client_type': client_types,
        'first_name': fnames,
        'last_name': lnames,
        'age': ages,
        'gender': genders,
        'country': countries,
        'region': regions,
        'purpose': purposes,
        'loan_applied': loans,
        'referral_channel': channels,
        'satisfaction': satisfactions
    })
    
    # 2. Generate 10,000 properties
    property_ids = list(range(1001, 11001))
    towers = [int((pid - 1001) // 1000 + 1) for pid in property_ids]
    units = [int((pid - 1001) % 1000 + 1) for pid in property_ids]
    categories = np.random.choice(['Apartment', 'Office'], size=10000, p=[0.85, 0.15])
    areas = np.random.randint(500, 2500, size=10000)
    prices = areas * np.random.uniform(180, 320, size=10000)
    
    client_refs = [""] * 10000
    statuses = ["Available"] * 10000
    
    sold_indices = np.random.choice(range(10000), size=1900, replace=False)
    for idx, client_idx in zip(sold_indices, np.random.randint(0, 2000, size=1900)):
        client_refs[idx] = f"C{client_idx+1:04d}"
        statuses[idx] = "Sold"
        
    properties = pd.DataFrame({
        'listing_id': property_ids,
        'tower_number': towers,
        'unit_number': units,
        'unit_category': categories,
        'floor_area_sqft': areas,
        'sale_price': prices,
        'listing_status': statuses,
        'client_ref': client_refs
    })
    
    return buyers, properties

buyers_df, properties_df = load_data()

# Pre-merge to map actual sales onto buyers
sold_properties = properties_df[properties_df['listing_status'] == 'Sold']
merged_df = pd.merge(buyers_df, sold_properties, left_on='client_id', right_on='client_ref', how='left')
merged_df['property_value'] = merged_df['sale_price'].fillna(0)

# Display KPI Overview cards at top
st.subheader("📊 Synchronized Database KPI Summary")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("Total Clients Registered", f"{len(buyers_df):,}")
with kpi_col2:
    st.metric("Total Properties Listed", f"{len(properties_df):,}")
with kpi_col3:
    st.metric("Properties Sold", f"{len(sold_properties):,}")
with kpi_col4:
    st.metric("Total Sales Volume", f"${sold_properties['sale_price'].sum():,.2f}")

# Sidebar Filters
st.sidebar.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?q=80&w=200", caption="Parcl Research Co.")
st.sidebar.header("🎯 Data Filters")
f_country = st.sidebar.selectbox("Country of Origin", ["All"] + list(buyers_df['country'].unique()))
f_purpose = st.sidebar.selectbox("Investment Purpose", ["All"] + list(buyers_df['purpose'].unique()))

filtered_df = merged_df.copy()
if f_country != "All":
    filtered_df = filtered_df[filtered_df['country'] == f_country]
if f_purpose != "All":
    filtered_df = filtered_df[filtered_df['purpose'] == f_purpose]

# ML Clustering block
st.header("🤖 Automatic ML Buyer Profiler (K-Means)")
k_clusters = st.slider("Select Target Segment Count (K)", 2, 8, 4)

# Robust clustering check to prevent ValueError: n_samples should be >= n_clusters
if len(filtered_df) < k_clusters:
    st.warning(f"⚠️ **Not enough data points selected!** Only **{len(filtered_df)}** buyers match your active filters (Country: '{f_country}', Purpose: '{f_purpose}'). Running a K-Means algorithm with **K = {k_clusters}** requires at least as many data points. Please reduce K or select broader filters.")
    filtered_df['cluster'] = "General Cluster"
else:
    features = filtered_df[['age', 'satisfaction', 'property_value']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    filtered_df['cluster_id'] = kmeans.fit_predict(scaled_features)
    
    # Calculate centroids
    centroids_temp = filtered_df.groupby('cluster_id')[['age', 'property_value', 'satisfaction']].mean()
    
    # Sort centroids by mean property_value descending to establish rank
    sorted_centroid_idx = centroids_temp.sort_values(by='property_value', ascending=False).index.tolist()
    
    # Beautiful, professional corporate buyer personas (C1 to C8 sequentially)
    personas = [
        "C1: Luxury Portfolio Investors",
        "C2: Global High-Net-Worth Investors",
        "C3: Institutional/Corporate Buyers",
        "C4: Strategic Mid-Market Buyers",
        "C5: High-Intent First-Time Buyers",
        "C6: Active High-Stress Buyers",
        "C7: Entry-Level Speculative Inquirers",
        "C8: Passive Micro-Value Buyers"
    ]
    
    cluster_map = {}
    for rank, cl_id in enumerate(sorted_centroid_idx):
        # Fallback security check
        if rank < len(personas):
            name = personas[rank]
        else:
            name = f"Cluster Persona {rank + 1}"
        cluster_map[cl_id] = name
        
    filtered_df['cluster'] = filtered_df['cluster_id'].map(cluster_map)

# Sidebar Expander for Centroids vs Global Averages
global_age = filtered_df['age'].mean()
global_val = filtered_df['property_value'].mean()
global_sat = filtered_df['satisfaction'].mean()

with st.sidebar.expander("📊 Segment vs Global Centroids", expanded=True):
    st.markdown("### 🎯 Global Filters Baseline")
    st.markdown(f"**Mean Age:** {global_age:.1f} yr\n"
                f"**Avg Value:** ${global_val:,.2f}\n"
                f"**Satisfaction:** {global_sat:.1f}/5")
    st.markdown("---")
    
    if "cluster" in filtered_df.columns and filtered_df['cluster'].iloc[0] != "General Cluster":
        centroids = filtered_df.groupby('cluster')[['age', 'property_value', 'satisfaction']].mean()
        for idx, row in centroids.iterrows():
            st.markdown(f"**🟢 {idx}**")
            age_dev = row['age'] - global_age
            val_dev = row['property_value'] - global_val
            sat_dev = row['satisfaction'] - global_sat
            
            st.markdown(
                f"- **Age:** {row['age']:.1f} yr ({'+' if age_dev >= 0 else ''}{age_dev:.1f} yr)\n"
                f"- **Value:** ${row['property_value']:,.2f} ({'+' if val_dev >= 0 else ''}${val_dev:,.2f})\n"
                f"- **Rating:** {row['satisfaction']:.1f}/5 ({'+' if sat_dev >= 0 else ''}{sat_dev:.1f})"
            )
            st.markdown(" ")
    else:
        st.info("Insufficient data points for robust segmentation comparison.")

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(
        filtered_df, x="age", y="property_value", color="cluster",
        size=np.where(filtered_df['property_value'] > 0, filtered_df['satisfaction'], 1),
        hover_data=['client_id', 'country', 'purpose'],
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
