# 03_Timeline.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_summary, load_full_results, get_application_data, filter_dataframe
from utils.visualizations import create_timeline_chart

# Page configuration
st.set_page_config(
    page_title="Timeline | Graphene Applications Dashboard",
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
st.markdown("<h1 class='page-header'>Graphene Applications Timeline</h1>", unsafe_allow_html=True)
st.write("This page explores the temporal development of graphene applications.")

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
        value=(2000, max_year)
    )

# Apply filters
filtered_df = filter_dataframe(app_df, selected_sources, year_range)

# General publications timeline
st.header("Publications Timeline")
st.plotly_chart(create_timeline_chart(summary), use_container_width=True)

# Timeline for top categories
st.header("Timeline by Category")

# Group by year and category, count occurrences
category_trends = filtered_df.groupby(['year', 'category']).size().reset_index(name='count')

# Create line chart
fig = px.line(
    category_trends,
    x='year',
    y='count',
    color='category',
    markers=True,
    title="Growth Trends by Category",
    labels={'count': 'Number of Mentions', 'year': 'Year'}
)

# Add vertical line for Nobel Prize
fig.add_vline(
    x=2010, 
    line_dash="dash", 
    line_color="red",
    annotation_text="Nobel Prize",
    annotation_position="top right"
)

st.plotly_chart(fig, use_container_width=True)

# Timeline heatmap - Year vs Category
st.header("Year vs Category Heatmap")

# Create pivot table for heatmap
category_year_pivot = filtered_df.pivot_table(
    index='year', 
    columns='category',
    values='application',
    aggfunc='count',
    fill_value=0
)

# Create heatmap
fig = px.imshow(
    category_year_pivot,
    labels=dict(x="Category", y="Year", color="Count"),
    x=category_year_pivot.columns,
    y=category_year_pivot.index,
    color_continuous_scale="viridis",
    aspect="auto",
    title="Category Distribution Over Time"
)

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Cumulative growth of applications
st.header("Cumulative Growth of Applications")

# Calculate cumulative growth
year_counts = filtered_df.groupby('year').size().reset_index(name='count')
year_counts = year_counts.sort_values('year')
year_counts['cumulative'] = year_counts['count'].cumsum()

# Create line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=year_counts['year'],
    y=year_counts['cumulative'],
    mode='lines+markers',
    name='Cumulative Growth',
    line=dict(color='royalblue', width=3),
    fill='tozeroy',
))

fig.add_trace(go.Scatter(
    x=year_counts['year'],
    y=year_counts['count'],
    mode='lines+markers',
    name='Annual Count',
    line=dict(color='firebrick', width=2),
))

# Add vertical line for Nobel Prize
fig.add_vline(
    x=2010, 
    line_dash="dash", 
    line_color="red",
    annotation_text="Nobel Prize",
    annotation_position="top right"
)

fig.update_layout(
    title="Cumulative Growth of Graphene Applications",
    xaxis_title="Year",
    yaxis_title="Number of Applications",
    legend_title="Metric",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# Key milestones in graphene research
st.header("Key Milestones in Graphene Research")

milestones = [
    {"year": 2004, "event": "Graphene first isolated at Manchester University", "researchers": "Geim and Novoselov"},
    {"year": 2005, "event": "Electrical properties of graphene demonstrated", "researchers": "Manchester Group"},
    {"year": 2008, "event": "Graphene CMOS integrated circuit demonstrated", "researchers": "IBM"},
    {"year": 2010, "event": "Nobel Prize in Physics awarded for graphene research", "researchers": "Geim and Novoselov"},
    {"year": 2013, "event": "EU launches €1B Graphene Flagship initiative", "researchers": "European Commission"},
    {"year": 2017, "event": "First commercial graphene-enhanced smartphone", "researchers": "Industry"},
    {"year": 2019, "event": "Large-scale CVD graphene production reaches commercial viability", "researchers": "Industry"}
]

# Convert to DataFrame
milestones_df = pd.DataFrame(milestones)

# Create interactive table
st.dataframe(milestones_df, use_container_width=True)