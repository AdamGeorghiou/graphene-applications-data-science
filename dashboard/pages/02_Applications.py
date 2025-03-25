import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_summary, load_full_results, get_application_data, filter_dataframe
from utils.visualizations import create_top_applications_chart

# Page configuration
st.set_page_config(
    page_title="Applications | Graphene Applications Dashboard",
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
st.markdown("<h1 class='page-header'>Graphene Applications Analysis</h1>", unsafe_allow_html=True)
st.write("This page focuses on detailed analysis of specific graphene applications.")

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
    
    category_filter = st.multiselect(
        "Application Categories",
        options=list(summary['application_category_distribution'].keys()),
        default=[]
    )

# Apply filters
filtered_df = filter_dataframe(app_df, selected_sources, year_range)

# Apply category filter if selected
if category_filter:
    filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]

# Top applications
st.header("Top Applications")
st.plotly_chart(create_top_applications_chart(summary), use_container_width=True)

# Application growth trends
st.header("Application Growth Trends")

# Group by year and application, count occurrences
app_trends = filtered_df.groupby(['year', 'application']).size().reset_index(name='count')

# Get top 5 applications
top_apps = app_trends.groupby('application')['count'].sum().sort_values(ascending=False).head(5).index.tolist()

# Filter to only include top 5 applications
app_trends_top = app_trends[app_trends['application'].isin(top_apps)]

# Create line chart
fig = px.line(
    app_trends_top,
    x='year',
    y='count',
    color='application',
    markers=True,
    title="Growth Trends of Top 5 Applications",
    labels={'count': 'Number of Mentions', 'year': 'Year'}
)
st.plotly_chart(fig, use_container_width=True)

# Application by category breakdown
st.header("Applications by Category")

# Create a DataFrame with counts by category and application
category_app_counts = filtered_df.groupby(['category', 'application']).size().reset_index(name='count')
category_app_counts = category_app_counts.sort_values(['category', 'count'], ascending=[True, False])

# For each category, show top 5 applications
categories = category_app_counts['category'].unique()

for i in range(0, len(categories), 2):
    cols = st.columns(2)
    
    for j in range(2):
        if i + j < len(categories):
            category = categories[i + j]
            with cols[j]:
                st.subheader(category)
                
                # Get top 5 applications for this category
                top_category_apps = category_app_counts[category_app_counts['category'] == category].head(5)
                
                fig = px.bar(
                    top_category_apps,
                    y='application',
                    x='count',
                    orientation='h',
                    color='count',
                    color_continuous_scale='viridis',
                    labels={'count': 'Number of Mentions', 'application': 'Application'},
                    title=f"Top Applications in {category}"
                )
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)