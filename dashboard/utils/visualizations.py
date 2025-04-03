#visualizations.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_category_chart(summary, app_categories):
    """Create sunburst chart for application categories"""
    categories = summary['application_category_distribution']
    top_apps = summary['top_applications']
    
    sunburst_data = []
    
    # Add root
    sunburst_data.append({
        'ids': "Graphene",
        'labels': "Graphene",
        'parents': "",
        'values': summary['total_applications_found']
    })
    
    # Add categories
    for category, count in categories.items():
        sunburst_data.append({
            'ids': category,
            'labels': category,
            'parents': "Graphene",
            'values': count
        })
    
    # Add top applications
    for app, count in top_apps.items():
        # Find which category this app belongs to using the passed app_categories
        for doc_app in app_categories:
            if doc_app['application'] == app:
                category = doc_app['category']
                break
        else:
            category = "Other"  # Fallback category if not found
        
        sunburst_data.append({
            'ids': app,
            'labels': app,
            'parents': category,
            'values': count
        })
    
    df = pd.DataFrame(sunburst_data)
    
    fig = px.sunburst(
        df, 
        ids='ids',
        names='labels',
        parents='parents',
        values='values',
        color_discrete_sequence=px.colors.qualitative.G10,
        title="Graphene Applications by Category"
    )
    
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=600)
    return fig

def create_top_applications_chart(summary):
    """Create horizontal bar chart for top applications"""
    # Get top applications data
    top_apps = summary['top_applications']
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Application': list(top_apps.keys()),
        'Count': list(top_apps.values())
    }).sort_values('Count', ascending=True)
    
    # Create bar chart
    fig = px.bar(
        df.tail(15),  # Get last 15 rows (highest counts)
        y='Application',
        x='Count',
        orientation='h',
        color='Count',
        color_continuous_scale='viridis',
        title="Top Graphene Applications by Mention Count"
    )
    
    fig.update_layout(
        xaxis_title="Number of Mentions",
        yaxis_title="",
        height=600,
    )
    
    return fig

def create_timeline_chart(summary):
    """Create timeline chart of applications development"""
    # Extract year distribution
    year_data = summary['year_distribution']
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Year': [int(year) for year in year_data.keys() if year.isdigit()],
        'Count': [count for year, count in year_data.items() if year.isdigit()]
    }).sort_values('Year')
    
    # Filter out years before 2000 for clearer visualization
    df = df[df['Year'] >= 2000]
    
    # Create line chart
    fig = px.line(
        df,
        x='Year',
        y='Count',
        markers=True,
        title="Graphene Applications Publications Timeline"
    )
    
    # Add vertical line for Nobel Prize (2010)
    fig.add_vline(
        x=2010, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Nobel Prize",
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        height=500,
    )
    
    return fig

def create_trl_heatmap(df):
    """Create heatmap of TRL levels by application type"""
    # Group by application and TRL
    trl_matrix = df.pivot_table(
        index='application', 
        columns='trl',
        values='id',
        aggfunc='count',
        fill_value=0
    )
    
    # Sort by total count
    trl_matrix['total'] = trl_matrix.sum(axis=1)
    trl_matrix = trl_matrix.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Keep only top 15 applications
    trl_matrix = trl_matrix.head(15)
    
    # Create heatmap
    fig = px.imshow(
        trl_matrix,
        labels=dict(x="TRL Level", y="Application", color="Count"),
        x=[str(i) for i in trl_matrix.columns],
        y=trl_matrix.index,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Technology Readiness Levels by Application"
    )
    
    fig.update_layout(
        height=600,
    )
    
    return fig

def create_fabrication_methods_chart(summary):
    """Create bar chart for fabrication methods"""
    # Extract fabrication methods
    fab_methods = summary['fabrication_methods']
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Method': list(fab_methods.keys()),
        'Count': list(fab_methods.values())
    }).sort_values('Count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Method',
        y='Count',
        color='Count',
        color_continuous_scale='viridis',
        title="Graphene Fabrication Methods"
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Number of Mentions",
        xaxis_tickangle=-45,
        height=500,
    )
    
    return fig

def create_research_type_pie(summary):
    """Create pie chart for research type distribution"""
    # Extract research type distribution
    research_types = summary['research_type_distribution']
    
    # Create pie chart
    fig = px.pie(
        values=list(research_types.values()),
        names=list(research_types.keys()),
        title="Research Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(
        height=400,
    )
    
    return fig

def get_category_for_app(app_name):
    """Fallback function to determine category for an application"""
    # Map applications to categories based on keywords
    category_keywords = {
        'Electronics': ['electronic', 'transistor', 'circuit', 'sensor', 'display', 'electrode'],
        'Energy': ['battery', 'supercapacitor', 'solar', 'fuel cell', 'energy'],
        'Materials': ['composite', 'coating', 'membrane', 'filter', 'barrier', 'reinforcement'],
        'Chemical': ['catalyst', 'oxidation', 'reduction'],
        'Biomedical': ['drug', 'tissue', 'bio', 'cellular', 'antibacterial'],
        'Environmental': ['water', 'gas', 'pollution', 'environmental', 'purification']
    }
    
    app_lower = app_name.lower()
    for category, keywords in category_keywords.items():
        if any(keyword in app_lower for keyword in keywords):
            return category
    
    return "Other"