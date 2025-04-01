import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Page configuration - must come first
st.set_page_config(
    page_title="Graphene Applications Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Load data directly without relying on utils
@st.cache_data(ttl=3600) # Cache data for 1 hour (adjust as needed)
def load_data():
    """
    Load NLP results, trying local path first and falling back to GitHub.
    Caches the result for faster performance.
    """
    import requests # Make sure requests is imported

    summary = None
    full_results = None
    app_df = pd.DataFrame() # Initialize empty DataFrame

    # --- Try Loading Locally ---
    try:
        # Construct local path relative to this script file
        script_dir = os.path.dirname(__file__)
        # Assuming app.py is in the root of the project, or adjust as needed
        # If app.py is in a 'src' folder, use os.path.join(script_dir, "..")
        project_root = script_dir # Change if app.py is not in project root
        nlp_results_dir = os.path.join(project_root, "data", "nlp_results")

        # Check if the directory and files exist locally
        summary_path = os.path.join(nlp_results_dir, "enhanced_nlp_summary.json")
        results_path = os.path.join(nlp_results_dir, "nlp_results.json")

        if os.path.exists(summary_path) and os.path.exists(results_path):
            st.write("Attempting to load data from local files...") # Info message
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            with open(results_path, 'r', encoding='utf-8') as f:
                full_results = json.load(f)

            st.success("✅ Loaded data successfully from local files.")
        else:
             # Raise error to trigger the except block if files aren't found locally
            raise FileNotFoundError("Local data files not found.")

    except Exception as e:
        st.warning(f"⚠️ Local loading failed ({e}). Attempting to load from GitHub...")

        # --- Fallback to Loading from GitHub ---
        # !!! IMPORTANT: Confirm GitHub username, repo name, and branch !!!
        github_username = "AdamGeorghiou"
        github_repo = "graphene-applications-data-science"
        github_branch = "main" # Or whichever branch holds the data
        github_path = "data/nlp_results" # Path within the repo

        base_url = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/{github_branch}/{github_path}"

        summary_url = f"{base_url}/enhanced_nlp_summary.json"
        results_url = f"{base_url}/nlp_results.json"

        try:
            # Load summary from GitHub
            summary_response = requests.get(summary_url)
            summary_response.raise_for_status() # Raise exception for bad status codes (like 404)
            summary = json.loads(summary_response.text)

            # Load full results from GitHub
            results_response = requests.get(results_url)
            results_response.raise_for_status() # Raise exception for bad status codes
            full_results = json.loads(results_response.text)

            st.success("✅ Loaded data successfully from GitHub.")

        except requests.exceptions.RequestException as github_e:
            st.error(f"❌ Failed to load data from GitHub: {github_e}")
            # Return empty data on failure
            return None, None, pd.DataFrame()
        except json.JSONDecodeError as json_e:
            st.error(f"❌ Failed to parse JSON data from GitHub: {json_e}")
            return None, None, pd.DataFrame()

    # --- Process Data (only if loaded successfully) ---
    if full_results:
        try:
            # Call your existing function to process the results
            app_df = get_application_data(full_results)
            # st.write("Processed application data into DataFrame.") # Optional info message
        except Exception as proc_e:
            st.error(f"❌ Failed to process loaded data: {proc_e}")
            app_df = pd.DataFrame() # Return empty if processing fails

    # Ensure we return even if loading/processing failed partially
    # If summary failed to load but results loaded, summary will be None
    return summary, full_results, app_df

def get_application_data(results):
    """Extract application data from results for visualization"""
    app_data = []
    for doc in results:
        source = doc.get('source', 'Unknown')
        year = None
        if 'publication_info' in doc and 'year' in doc['publication_info']:
            year_val = doc['publication_info']['year']
            if isinstance(year_val, (int, float)) and not pd.isna(year_val):
                year = int(year_val)
            elif isinstance(year_val, str) and year_val.isdigit():
                year = int(year_val)
        
        trl = doc.get('trl_assessment', {}).get('trl_score', 0)
        
        for app in doc.get('applications', []):
            app_name = app.get('application', 'Unknown')
            category = app.get('category', 'Unknown')
            
            app_data.append({
                'id': doc.get('id', ''),
                'source': source,
                'year': year,
                'application': app_name,
                'category': category,
                'trl': trl
            })
    
    return pd.DataFrame(app_data)

def filter_dataframe(df, sources=None, year_range=None):
    """Filter a dataframe based on sources and year range"""
    filtered_df = df.copy()
    
    if sources and len(sources) > 0:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    if year_range and len(year_range) == 2:
        min_year, max_year = year_range
        filtered_df = filtered_df[
            (filtered_df['year'] >= min_year) & 
            (filtered_df['year'] <= max_year)
        ]
    
    return filtered_df

# Visualization functions - directly included in app.py
def create_category_chart(summary):
    """Create sunburst chart for application categories"""
    # Extract category data
    categories = summary['application_category_distribution']
    
    # Create data for top applications in each category
    top_apps = summary['top_applications']
    
    # Prepare data for sunburst chart
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
    
    # Add top applications with simple category assignment
    for app, count in top_apps.items():
        # Simple category determination
        category = get_category_for_app(app)
        
        sunburst_data.append({
            'ids': app,
            'labels': app,
            'parents': category,
            'values': count
        })
    
    # Convert to DataFrame for plotly
    df = pd.DataFrame(sunburst_data)
    
    # Create sunburst chart
    fig = px.sunburst(
        df, 
        ids='ids',
        names='labels',
        parents='parents',
        values='values',
        color_discrete_sequence=px.colors.qualitative.G10,
        title="Graphene Applications by Category"
    )
    
    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        height=600,
    )
    
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
    """Simple function to determine category for an application"""
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

# Load the data
summary, full_results, app_df = load_data()

# Extract unique sources for filters
all_sources = app_df['source'].unique().tolist()

# Determine year range for filters
min_year = int(app_df['year'].min()) if not pd.isna(app_df['year'].min()) else 2000
max_year = int(app_df['year'].max()) if not pd.isna(app_df['year'].max()) else 2023

# Sidebar filters
st.sidebar.title("Filters")
selected_sources = st.sidebar.multiselect(
    "Data Sources",
    options=all_sources,
    default=all_sources
)

year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year, 
    max_value=max_year,
    value=(2010, max_year)
)

# Apply filters
filtered_df = filter_dataframe(app_df, selected_sources, year_range)

# Main dashboard
st.markdown("<h1 class='main-header'>Graphene Applications Dashboard</h1>", unsafe_allow_html=True)
st.write("This dashboard visualizes the results of analyzing graphene applications using data science techniques.")

# KPI metrics row
st.markdown("<h2 class='subheader'>Key Metrics</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Documents Analyzed", f"{summary['total_documents']:,}")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Applications Found", f"{summary['total_applications_found']:,}")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Unique Applications", summary['unique_applications'])
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Data Sources", len(all_sources))
    st.markdown("</div>", unsafe_allow_html=True)

# Main charts section
st.markdown("<h2 class='subheader'>Application Overview</h2>", unsafe_allow_html=True)

# Create the tabs
tab1, tab2, tab3 = st.tabs(["Categories", "Top Applications", "Research Type"])

with tab1:
    try:
        st.plotly_chart(create_category_chart(summary), use_container_width=True)
    except Exception as e:
        st.error(f"Error creating category chart: {str(e)}")
    
with tab2:
    try:
        st.plotly_chart(create_top_applications_chart(summary), use_container_width=True)
    except Exception as e:
        st.error(f"Error creating top applications chart: {str(e)}")
    
with tab3:
    try:
        st.plotly_chart(create_research_type_pie(summary), use_container_width=True)
    except Exception as e:
        st.error(f"Error creating research type chart: {str(e)}")

# Timeline section
st.markdown("<h2 class='subheader'>Publication Timeline</h2>", unsafe_allow_html=True)
try:
    st.plotly_chart(create_timeline_chart(summary), use_container_width=True)
    st.caption("Note: The graph shows a significant increase in graphene applications after the 2010 Nobel Prize.")
except Exception as e:
    st.error(f"Error creating timeline chart: {str(e)}")

# Applications by source visualization
st.markdown("<h2 class='subheader'>Applications by Source</h2>", unsafe_allow_html=True)
try:
    # Create bar chart of applications by source
    source_counts = filtered_df['source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    
    fig = px.bar(
        source_counts,
        x='Source',
        y='Count',
        color='Count',
        color_continuous_scale='viridis',
        title="Applications by Data Source"
    )
    
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error creating source chart: {str(e)}")

# Dashboard information
st.markdown("<h2 class='subheader'>About this Dashboard</h2>", unsafe_allow_html=True)
st.write("""
This dashboard visualizes the results of a data science project analyzing graphene applications across various industries.
The data was collected from multiple sources including academic papers, patents, and research publications.

The analysis involved:
- Collecting data from Google Patents, Scopus, arXiv, and Semantic Scholar
- Cleaning and normalizing the data
- Applying NLP techniques to extract applications, fabrication methods, and technical details
- Categorizing applications and identifying trends

You can use the filters in the sidebar to explore the data by source and year range.
""")

# Footer
st.markdown("---")
st.markdown("Data last updated: February 2023 | Dashboard created with Streamlit")