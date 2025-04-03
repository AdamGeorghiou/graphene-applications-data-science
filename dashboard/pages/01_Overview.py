# 01_Overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_summary, load_full_results, get_application_data, filter_dataframe
from utils.visualizations import create_category_chart, create_top_applications_chart, create_research_type_pie

# Page configuration
st.set_page_config(
    page_title="Overview | Graphene Applications Dashboard",
    page_icon="📊",
    layout="wide",
)

# Add custom CSS
st.markdown("""
    <style>
    .page-header {
        font-size: 2.3rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    summary = load_summary()
    full_results = load_full_results()
    app_df = get_application_data(full_results)
    return summary, full_results, app_df

summary, full_results, app_df = load_data()

# Page header
st.markdown("<h1 class='page-header'>Graphene Applications Overview</h1>", unsafe_allow_html=True)
st.write("This page provides a comprehensive overview of graphene applications identified in the dataset.")

# Extract unique sources for filters
all_sources = app_df['source'].unique().tolist()

# Determine year range for filters
min_year = int(app_df['year'].min()) if not pd.isna(app_df['year'].min()) else 2000
max_year = int(app_df['year'].max()) if not pd.isna(app_df['year'].max()) else 2023

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    selected_sources = st.multiselect(
        "Data Sources",
        options=all_sources,
        default=all_sources
    )
    year_range = st.slider(
        "Year Range",
        min_value=min_year, 
        max_value=max_year,
        value=(2010, max_year)
    )

# Apply filters
filtered_df = filter_dataframe(app_df, selected_sources, year_range)
app_categories = filtered_df.drop_duplicates(subset=['application']).to_dict('records')

# Overview metrics
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Documents", f"{summary['total_documents']:,}")
with col2:
    st.metric("Total Applications", f"{summary['total_applications_found']:,}")
with col3:
    st.metric("Unique Applications", summary['unique_applications'])
with col4:
    filtered_count = len(filtered_df)
    st.metric("Filtered Applications", f"{filtered_count:,}")

# Application categories (only one call)
st.header("Application Categories")
st.plotly_chart(create_category_chart(summary, app_categories), use_container_width=True)

# Source distribution
st.header("Source Distribution")
source_counts = filtered_df['source'].value_counts().reset_index()
source_counts.columns = ['Source', 'Count']

fig = px.pie(
    source_counts, 
    values='Count', 
    names='Source',
    title="Distribution by Source",
    hole=0.4,
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)

# Research Type Distribution
st.header("Research Type Distribution")
st.plotly_chart(create_research_type_pie(summary), use_container_width=True)
