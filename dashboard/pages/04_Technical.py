import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.data_loader import load_summary, load_full_results, get_application_data, get_fabrication_data, filter_dataframe
from utils.visualizations import create_trl_heatmap, create_fabrication_methods_chart

# Page configuration
st.set_page_config(
    page_title="Technical Analysis | Graphene Applications Dashboard",
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
    fab_df = get_fabrication_data(full_results)
    return summary, full_results, app_df, fab_df

summary, full_results, app_df, fab_df = load_data()

# Page header
st.markdown("<h1 class='page-header'>Technical Analysis of Graphene Applications</h1>", unsafe_allow_html=True)
st.write("This page focuses on the technical aspects of graphene applications, including fabrication methods and TRL assessment.")

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
filtered_app_df = filter_dataframe(app_df, selected_sources, year_range)
filtered_fab_df = filter_dataframe(fab_df, selected_sources, year_range)

# TRL Distribution
st.header("Technology Readiness Level (TRL) Distribution")
trl_counts = filtered_app_df['trl'].value_counts().sort_index()

fig = px.bar(
    x=trl_counts.index.astype(str),
    y=trl_counts.values,
    color=trl_counts.values,
    color_continuous_scale='viridis',
    labels={'x': 'TRL Level', 'y': 'Count'},
    title="TRL Distribution"
)

fig.update_layout(
    xaxis_title="TRL Level",
    yaxis_title="Number of Applications",
    height=400,
)

st.plotly_chart(fig, use_container_width=True)

# Description of TRL levels
with st.expander("About TRL Levels"):
    st.markdown("""
    **Technology Readiness Levels (TRL)** are a method for estimating the maturity of technologies:
    
    - **TRL 1**: Basic principles observed and reported
    - **TRL 2**: Technology concept and/or application formulated
    - **TRL 3**: Analytical and experimental critical function and/or proof of concept
    - **TRL 4**: Component and/or validation in laboratory environment
    - **TRL 5**: Component and/or validation in relevant environment
    - **TRL 6**: System/subsystem model or prototype demonstration in relevant environment
    - **TRL 7**: System prototype demonstration in operational environment
    - **TRL 8**: Actual system completed and qualified through test and demonstration
    - **TRL 9**: Actual system proven in operational environment
    
    The skew toward lower TRL levels in graphene applications indicates many technologies are still in early development stages.
    """)

# TRL Heatmap
st.header("TRL by Application Type")
st.plotly_chart(create_trl_heatmap(filtered_app_df), use_container_width=True)

# Fabrication methods
st.header("Fabrication Methods")
st.plotly_chart(create_fabrication_methods_chart(summary), use_container_width=True)

# Fabrication methods timeline
if 'method' in filtered_fab_df.columns and not filtered_fab_df.empty:
    st.header("Fabrication Methods Timeline")
    
    # Get top 5 methods
    top_methods = filtered_fab_df['method'].value_counts().head(5).index.tolist()
    
    # Filter to only top methods for clearer visualization
    top_methods_df = filtered_fab_df[filtered_fab_df['method'].isin(top_methods)]
    
    # Group by year and method
    method_trends = top_methods_df.groupby(['year', 'method']).size().reset_index(name='count')
    
    # Create line chart
    fig = px.line(
        method_trends,
        x='year',
        y='count',
        color='method',
        markers=True,
        title="Evolution of Top Fabrication Methods",
        labels={'count': 'Number of Mentions', 'year': 'Year'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Relationship between fabrication methods and applications
st.header("Fabrication Method to Application Relationships")

# Create a joint dataframe connecting methods and applications
if not filtered_fab_df.empty and not filtered_app_df.empty:
    # Merge on id and year to connect fabrication methods with applications
    fab_app_df = pd.merge(
        filtered_fab_df, 
        filtered_app_df, 
        on=['id', 'source', 'year'],
        how='inner'
    )
    
    # Group by method and application
    fab_app_counts = fab_app_df.groupby(['method', 'application']).size().reset_index(name='count')
    
    # Filter for top relationships
    fab_app_counts = fab_app_counts.sort_values('count', ascending=False).head(15)
    
    # Create horizontal bar chart
    fig = px.bar(
        fab_app_counts,
        y='method',
        x='count',
        color='application',
        orientation='h',
        title="Top Fabrication Method-Application Combinations",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Insufficient data to analyze fabrication method to application relationships.")

# Performance metrics analysis
st.header("Performance Metrics Analysis")

# Extract performance metrics from summary
perf_metrics = summary['performance_metrics_summary']

# Prepare data for visualization
perf_data = []
for app, metrics in perf_metrics.items():
    for metric, values in metrics.items():
        if isinstance(values, dict) and 'avg' in values:
            perf_data.append({
                'application': app,
                'metric': metric,
                'avg_value': values['avg'],
                'min_value': values['min'],
                'max_value': values['max'],
                'count': values['count']
            })

if perf_data:
    # Convert to DataFrame
    perf_df = pd.DataFrame(perf_data)
    
    # Filter for applications with sufficient data
    perf_df = perf_df[perf_df['count'] >= 3]
    
    if not perf_df.empty:
        # Create visualization for average values
        fig = px.bar(
            perf_df,
            x='application',
            y='avg_value',
            color='metric',
            barmode='group',
            labels={'avg_value': 'Average Value', 'application': 'Application'},
            title="Average Performance Metrics by Application",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create range chart for min/max values
        fig = go.Figure()
        
        for metric in perf_df['metric'].unique():
            metric_df = perf_df[perf_df['metric'] == metric]
            
            fig.add_trace(go.Scatter(
                x=metric_df['application'],
                y=metric_df['avg_value'],
                mode='markers',
                name=f'{metric} (avg)',
                marker=dict(size=8, color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=metric_df['application'],
                y=metric_df['min_value'],
                mode='lines',
                name=f'{metric} (min)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=metric_df['application'],
                y=metric_df['max_value'],
                mode='lines',
                name=f'{metric} (max)',
                fill='tonexty',
                line=dict(width=0),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Performance Metrics Range by Application",
            xaxis_title="Application",
            yaxis_title="Value Range",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient performance metrics data for visualization. Most applications have fewer than 3 data points.")
else:
    st.info("No performance metrics data available.")