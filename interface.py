import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import io
from functions import (get_summary, manual_cleaning, auto_clean, perform_eda, generate_insights, 
                       get_download_link, get_outlier_info, treat_outliers, treat_all_outliers)
from visualization import create_all_distributions, create_correlation_heatmap, create_correlation_scatter
from machinelearning import train_model
from llm_insights import enhance_insights_with_llm, generate_quick_summary

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="DataPulse: Automated EDA", layout="wide", initial_sidebar_state="expanded")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eda_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox div[data-baseweb="select"] > div, 
    .stFileUploader label {
        background-color: #ffffff;
        color: #1a3c6d !important;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #1a3c6d !important;
    }
    .stSelectbox div[data-baseweb="select"] > div[aria-expanded="true"],
    .stSelectbox div[data-baseweb="select"] > div:hover {
        background-color: #e6f3ff !important;
        border-color: #007bff !important;
    }
    .stSelectbox div[data-baseweb="select"] ul[role="listbox"] li[aria-selected="true"] {
        background-color: #cce5ff !important;
        color: #1a3c6d !important;
    }
    .stCheckbox div[role="checkbox"] {
        background-color: #ffffff;
        color: #1a3c6d !important;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stCheckbox div[role="checkbox"] > div {
        color: #1a3c6d !important;
    }
    .stCheckbox div[role="checkbox"][aria-checked="true"] {
        background-color: #e6f3ff !important;
        border-color: #007bff !important;
    }
    .stCheckbox div[role="checkbox"][aria-checked="true"] > div {
        background-color: #007bff !important;
        border-color: #007bff !important;
    }
    h1, h2, h3 {
        color: #1a3c6d;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTabs > div > button {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        color: #1a3c6d;
    }
    .stTabs > div > button:hover {
        background-color: #ced4da;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stSpinner > div {
        border-color: #007bff !important;
    }
    .ai-insights-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .insight-content {
        background-color: white;
        color: #1a3c6d;
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .outlier-info-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .plot-divider {
        margin: 30px 0;
        border-bottom: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
# DataPulse: Automated EDA
**Created by**: Rakesh Kapilavayi  
**About Me**:  
- **Role**: Aspiring Data Scientist  
- **Skills**: Python, SQL, Data Cleaning, EDA, Visualization (Plotly), Machine Learning (Scikit-learn), Streamlit  
- **Contact**:  
  - Email: rakeshkapilavayi978@gmail.com  
  - LinkedIn: [Rakesh Kapilavayi](https://www.linkedin.com/in/rakesh-kapilavayi-48b9a0342/)  
  - GitHub: [rakeshkapilavayi](https://github.com/rakeshkapilavayi)  

**Project Overview**:  
This app allows users to:  
- Upload CSV/Excel files for analysis  
- Perform manual and automated data cleaning  
- Conduct interactive EDA with Plotly visualizations  
- Detect and treat outliers  
- Train machine learning models (classification/regression)  
- 🤖 Generate enhanced insights with advanced analysis  
- Export cleaned datasets  
""", unsafe_allow_html=True)

# Set page title
st.title("DataPulse: Automated Exploratory Data Analysis")

# Helper function to download plotly figure
def download_plot(fig, filename):
    """Create a download button for plotly figure"""
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode('utf-8')
    return html_bytes

# Helper function to download model
def download_model(model, filename):
    """Create a download button for trained model"""
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()

# File upload
with st.container():
    st.markdown("### Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key="file_uploader")
    
    # Store the name of the uploaded file to detect changes
    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = None
    
    if uploaded_file is not None:
        # Check if a new file is uploaded
        if uploaded_file.name != st.session_state['last_uploaded_file']:
            # Reset session state
            st.session_state['df'] = None
            st.session_state['cleaned_df'] = None
            st.session_state['model'] = None
            st.session_state['features'] = None
            st.session_state['task_type'] = None
            st.session_state['label_encoder'] = None
            st.session_state['cleaning_report'] = None
            st.session_state['ml_report'] = None
            st.session_state['last_uploaded_file'] = uploaded_file.name
            logger.info(f"New dataset uploaded: {uploaded_file.name}. Session state reset.")
        
        try:
            with st.spinner("Loading dataset..."):
                # Load the dataset
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Initialize or update session state
                st.session_state['df'] = df
                st.session_state['cleaned_df'] = df.copy()
                logger.info(f"Dataset '{uploaded_file.name}' loaded successfully. Shape: {df.shape}")

                # Display dataset preview
                st.markdown("### Dataset Preview")
                st.dataframe(st.session_state['cleaned_df'].head(), width='stretch')

                # Create tabs
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                    "Summary", "Manual Cleaning", "Auto Cleaning", "Developer Console", 
                    "Visualizations", "Outliers", "Machine Learning", "📊 Insights"
                ])

                # Tab 1: Summary
                with tab1:
                    st.markdown("### Dataset Summary")
                    try:
                        summary = get_summary(st.session_state['cleaned_df'])
                        summary_df = pd.DataFrame({
                            "Column Name": summary['columns'],
                            "Data Type": [summary['dtypes'][col] for col in summary['columns']],
                            "Missing Values": [summary['missing_values'][col] for col in summary['columns']],
                            "Unique Values": [st.session_state['cleaned_df'][col].nunique() for col in summary['columns']]
                        })
                        
                        st.markdown("#### Column Information")
                        st.dataframe(
                            summary_df.style.set_properties(**{'text-align': 'left'}),
                            width="stretch"
                        )
                        st.markdown(f"**Total Rows**: {summary['shape'][0]}")
                        st.markdown(f"**Total Columns**: {summary['shape'][1]}")
                        st.markdown(f"**Duplicate Rows**: {summary['duplicates']}")
                        logger.info("Summary tab displayed successfully")
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        logger.error(f"Summary error: {e}")

                # Tab 2: Manual Cleaning
                with tab2:
                    st.markdown("### Manual Data Cleaning")
                    missing_actions = {}
                    missing_columns = st.session_state['cleaned_df'].columns[st.session_state['cleaned_df'].isnull().any()].tolist()
                    
                    if missing_columns:
                        st.markdown("#### Handle Missing Values")
                        for col in missing_columns:
                            with st.expander(f"Column: {col} (Missing: {st.session_state['cleaned_df'][col].isnull().sum()})"):
                                action = st.selectbox(
                                    f"Action for {col}",
                                    ["None", "Drop", "Mean", "Median", "Mode"],
                                    key=f"missing_{col}"
                                )
                                if action != "None":
                                    missing_actions[col] = action
                    else:
                        st.info("No missing values found in the dataset.")
                    
                    remove_duplicates = st.checkbox("Remove Duplicate Rows", key="remove_duplicates")
                    
                    if st.button("Apply Manual Cleaning", key="apply_manual"):
                        try:
                            with st.spinner("Applying manual cleaning..."):
                                st.session_state['cleaned_df'] = manual_cleaning(
                                    st.session_state['cleaned_df'], missing_actions, remove_duplicates
                                )
                                st.success("Manual cleaning applied successfully!")
                                st.markdown("### Cleaned Dataset Preview")
                                st.dataframe(st.session_state['cleaned_df'].head(), width="stretch")
                                logger.info("Manual cleaning applied successfully")
                        except Exception as e:
                            st.error(f"Error applying manual cleaning: {e}")
                            logger.error(f"Manual cleaning error: {e}")

                # Tab 3: Auto Cleaning
                with tab3:
                    st.markdown("### Automated Data Cleaning")
                    if st.button("Perform Auto Cleaning", key="auto_clean"):
                        try:
                            with st.spinner("Performing auto cleaning..."):
                                st.session_state['cleaned_df'], report = auto_clean(st.session_state['cleaned_df'])
                                st.session_state['cleaning_report'] = report
                                st.success("Auto cleaning completed!")
                                st.markdown(f"**New Shape**: {st.session_state['cleaned_df'].shape[0]} rows, {st.session_state['cleaned_df'].shape[1]} columns")
                                st.markdown("#### Cleaning Report")
                                st.write(f"- Missing Values Handled: {len(report['missing_handled'])} columns")
                                for col, method in report['missing_handled'].items():
                                    st.write(f"  - {col}: {method}")
                                st.write(f"- Duplicates Removed: {report['duplicates_removed']}")
                                st.write(f"- Outliers Capped: {len(report['outliers_capped'])} columns")
                                st.markdown("### Cleaned Dataset Preview")
                                st.dataframe(st.session_state['cleaned_df'].head(), width="stretch")
                                logger.info("Auto cleaning completed successfully")
                        except Exception as e:
                            st.error(f"Error during auto cleaning: {e}")
                            logger.error(f"Auto cleaning error: {e}")

                # Tab 4: Developer Console
                with tab4:
                    st.markdown("### 🧑‍💻 Developer Console — Custom Data Operations")
                    if 'cleaned_df' not in st.session_state or st.session_state['cleaned_df'] is None:
                        st.warning("Please upload and process a dataset first.")
                    else:
                        st.info("Write Python code using 'df' as your main dataset.")
                        
                        second_file = st.file_uploader("Upload Second Dataset (for merge/concat)", type=["csv", "xlsx"], key="second_uploader")
                        other_df = None
                        if second_file:
                            try:
                                if second_file.name.endswith('.csv'):
                                    other_df = pd.read_csv(second_file)
                                else:
                                    other_df = pd.read_excel(second_file)
                                st.success(f"Second dataset loaded: {other_df.shape[0]} rows, {other_df.shape[1]} columns")
                                st.dataframe(other_df.head(3), width="stretch")
                            except Exception as e:
                                st.error(f"Error loading second dataset: {e}")

                        user_code = st.text_area("Write your Python code here:", height=200, key="dev_code")

                        if st.button("Preview Changes", key="preview_code"):
                            if user_code.strip():
                                with st.spinner("Running code preview..."):
                                    try:
                                        df_preview = st.session_state['cleaned_df'].copy()
                                        safe_globals = {
                                            "__builtins__": {
                                                "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                                                "range": range, "list": list, "dict": dict, "str": str, "int": int, "float": float,
                                                "bool": bool, "set": set, "tuple": tuple
                                            }
                                        }
                                        safe_locals = {
                                            "df": df_preview,
                                            "pd": pd,
                                            "np": np,
                                            "other_df": other_df
                                        }
                                        output = []
                                        safe_locals["print"] = lambda *args: output.append(" ".join(map(str, args)))

                                        exec(user_code, safe_globals, safe_locals)
                                        df_preview = safe_locals.get("df", df_preview)

                                        st.markdown("#### Preview Output")
                                        if output:
                                            for line in output:
                                                st.code(line, language="text")
                                        
                                        st.markdown("#### Preview of Modified Dataset")
                                        st.dataframe(df_preview.head(), width="stretch")
                                        st.write(f"New Shape: {df_preview.shape[0]} rows, {df_preview.shape[1]} columns")

                                        st.session_state['dev_preview_df'] = df_preview
                                        st.session_state['dev_preview_output'] = output

                                        st.success("Preview ready! Check below to apply changes.")
                                        logger.info("Developer console code preview executed successfully")
                                    except Exception as e:
                                        st.error(f"Error running code: {str(e)}")
                                        logger.error(f"Developer console error: {str(e)}")
                            else:
                                st.warning("Please enter some code to preview.")

                        if 'dev_preview_df' in st.session_state:
                            if st.button("Apply Changes to Main Dataset", key="apply_dev_changes"):
                                st.session_state['cleaned_df'] = st.session_state['dev_preview_df']
                                st.success("Changes applied!")
                                logger.info("Developer console changes applied")
                                del st.session_state['dev_preview_df']
                                del st.session_state['dev_preview_output']

                        st.markdown("#### Export Modified Dataset")
                        st.markdown(
                            get_download_link(st.session_state['cleaned_df'], filename=f"dev_modified_{uploaded_file.name}"),
                            unsafe_allow_html=True
                        )

                # Tab 5: IMPROVED EDA with Custom Charts
                with tab5:
                    st.markdown("### 📊 Exploratory Data Analysis")
                    
                    # Create two sections: Auto Distributions and Custom Charts
                    eda_option = st.radio(
                        "Choose EDA Mode:",
                        ["Auto Distributions", "Custom Chart Builder"],
                        horizontal=True
                    )
                    
                    if eda_option == "Auto Distributions":
                        try:
                            # Generate all distributions
                            with st.spinner("Generating visualizations..."):
                                distributions = create_all_distributions(st.session_state['cleaned_df'])
                            
                            # Show numerical distributions
                            if distributions['numerical']:
                                st.markdown("#### 📈 Numerical Columns Distribution")
                                for idx, item in enumerate(distributions['numerical']):
                                    st.plotly_chart(item['figure'], width="stretch")
                                    # Download button for each plot
                                    st.download_button(
                                        label=f"📥 Download {item['column']} Plot",
                                        data=download_plot(item['figure'], f"{item['column']}_distribution.html"),
                                        file_name=f"{item['column']}_distribution.html",
                                        mime="text/html",
                                        key=f"download_num_{idx}"
                                    )
                                    st.markdown('<div class="plot-divider"></div>', unsafe_allow_html=True)
                            else:
                                st.info("No numerical columns found in the dataset.")
                            
                            # Show categorical distributions
                            if distributions['categorical']:
                                st.markdown("#### 📊 Categorical Columns Distribution")
                                for idx, item in enumerate(distributions['categorical']):
                                    st.plotly_chart(item['figure'], width="stretch")
                                    # Download button for each plot
                                    st.download_button(
                                        label=f"📥 Download {item['column']} Plot",
                                        data=download_plot(item['figure'], f"{item['column']}_distribution.html"),
                                        file_name=f"{item['column']}_distribution.html",
                                        mime="text/html",
                                        key=f"download_cat_{idx}"
                                    )
                                    st.markdown('<div class="plot-divider"></div>', unsafe_allow_html=True)
                            else:
                                st.info("No categorical columns found in the dataset.")
                            
                            # Correlation Analysis
                            num_cols = st.session_state['cleaned_df'].select_dtypes(include=['float64', 'int64']).columns
                            if len(num_cols) >= 2:
                                st.markdown("#### 🔗 Correlation Analysis")
                                
                                # Correlation Heatmap
                                heatmap = create_correlation_heatmap(st.session_state['cleaned_df'])
                                if heatmap:
                                    st.plotly_chart(heatmap, width="stretch")
                                    st.download_button(
                                        label="📥 Download Correlation Heatmap",
                                        data=download_plot(heatmap, "correlation_heatmap.html"),
                                        file_name="correlation_heatmap.html",
                                        mime="text/html",
                                        key="download_heatmap"
                                    )
                                    st.markdown('<div class="plot-divider"></div>', unsafe_allow_html=True)
                                
                                # Correlation Scatter
                                scatter = create_correlation_scatter(st.session_state['cleaned_df'])
                                if scatter:
                                    st.plotly_chart(scatter, width="stretch")
                                    st.download_button(
                                        label="📥 Download Correlation Scatter",
                                        data=download_plot(scatter, "correlation_scatter.html"),
                                        file_name="correlation_scatter.html",
                                        mime="text/html",
                                        key="download_scatter"
                                    )
                            
                            logger.info("EDA visualizations displayed successfully")
                        except Exception as e:
                            st.error(f"Error generating EDA visualizations: {e}")
                            logger.error(f"EDA error: {e}")
                    
                    else:  # Custom Chart Builder
                        st.markdown("#### 🎨 Custom Chart Builder")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            chart_type = st.selectbox(
                                "Select Chart Type",
                                ["Bar Chart", "Pie Chart", "Scatter Plot", "Line Chart", "Box Plot", "Violin Plot", "Histogram"]
                            )
                        
                        with col2:
                            all_columns = st.session_state['cleaned_df'].columns.tolist()
                            x_axis = st.selectbox("Select X-axis", ["None"] + all_columns, key="x_axis_custom")
                        
                        with col3:
                            y_axis = st.selectbox("Select Y-axis", ["None"] + all_columns, key="y_axis_custom")
                        
                        # Additional options for some chart types
                        color_by = None
                        if chart_type in ["Scatter Plot", "Bar Chart", "Line Chart"]:
                            color_by = st.selectbox("Color by (optional)", ["None"] + all_columns, key="color_by")
                            if color_by == "None":
                                color_by = None
                        
                        if st.button("Generate Custom Chart", key="generate_custom"):
                            try:
                                fig = None
                                
                                if chart_type == "Bar Chart" and x_axis != "None":
                                    if y_axis != "None":
                                        fig = px.bar(st.session_state['cleaned_df'], x=x_axis, y=y_axis, 
                                                    color=color_by, title=f"Bar Chart: {x_axis} vs {y_axis}")
                                    else:
                                        fig = px.histogram(st.session_state['cleaned_df'], x=x_axis, 
                                                          color=color_by, title=f"Bar Chart: {x_axis}")
                                
                                elif chart_type == "Pie Chart" and x_axis != "None":
                                    value_counts = st.session_state['cleaned_df'][x_axis].value_counts()
                                    fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                                title=f"Pie Chart: {x_axis}")
                                
                                elif chart_type == "Scatter Plot" and x_axis != "None" and y_axis != "None":
                                    fig = px.scatter(st.session_state['cleaned_df'], x=x_axis, y=y_axis, 
                                                    color=color_by, title=f"Scatter Plot: {x_axis} vs {y_axis}")
                                
                                elif chart_type == "Line Chart" and x_axis != "None" and y_axis != "None":
                                    fig = px.line(st.session_state['cleaned_df'], x=x_axis, y=y_axis, 
                                                 color=color_by, title=f"Line Chart: {x_axis} vs {y_axis}")
                                
                                elif chart_type == "Box Plot" and y_axis != "None":
                                    fig = px.box(st.session_state['cleaned_df'], y=y_axis, x=x_axis if x_axis != "None" else None,
                                                title=f"Box Plot: {y_axis}")
                                
                                elif chart_type == "Violin Plot" and y_axis != "None":
                                    fig = px.violin(st.session_state['cleaned_df'], y=y_axis, x=x_axis if x_axis != "None" else None,
                                                   title=f"Violin Plot: {y_axis}")
                                
                                elif chart_type == "Histogram" and x_axis != "None":
                                    fig = px.histogram(st.session_state['cleaned_df'], x=x_axis, 
                                                      title=f"Histogram: {x_axis}")
                                
                                if fig:
                                    st.plotly_chart(fig, width="stretch")
                                    st.download_button(
                                        label="📥 Download Custom Chart",
                                        data=download_plot(fig, "custom_chart.html"),
                                        file_name="custom_chart.html",
                                        mime="text/html",
                                        key="download_custom"
                                    )
                                else:
                                    st.warning("Please select appropriate axes for the chosen chart type.")
                                
                            except Exception as e:
                                st.error(f"Error generating custom chart: {e}")
                                logger.error(f"Custom chart error: {e}")

                # Tab 6: Outliers with Treatment
                with tab6:
                    st.markdown("### 🎯 Outlier Detection and Treatment")
                    num_columns = st.session_state['cleaned_df'].select_dtypes(include=['float64', 'int64']).columns
                    
                    if num_columns.size > 0:
                        st.markdown("#### 📦 Outlier Visualization")
                        
                        for col in num_columns:
                            # Box plot
                            fig = px.box(
                                st.session_state['cleaned_df'], 
                                y=col, 
                                title=f'Box Plot of {col}',
                                color_discrete_sequence=['#00CC96']
                            )
                            st.plotly_chart(fig, width="stretch")
                            
                            # Download button for box plot
                            st.download_button(
                                label=f"📥 Download {col} Box Plot",
                                data=download_plot(fig, f"{col}_boxplot.html"),
                                file_name=f"{col}_boxplot.html",
                                mime="text/html",
                                key=f"download_box_{col}"
                            )
                            
                            # Get outlier info
                            outlier_info = get_outlier_info(st.session_state['cleaned_df'], col)
                            
                            if outlier_info and outlier_info['outlier_count'] > 0:
                                # Display outlier information
                                st.markdown(f"""
                                <div class="outlier-info-box">
                                    <strong>⚠️ Outlier Information for {col}</strong><br>
                                    <ul>
                                        <li>Total Outliers: {outlier_info['outlier_count']} ({outlier_info['outlier_percentage']:.2f}%)</li>
                                        <li>IQR Bounds: [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}]</li>
                                        <li>Q1: {outlier_info['Q1']:.2f}, Q3: {outlier_info['Q3']:.2f}</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Outlier treatment section
                                with st.expander(f"🛠️ Treat Outliers in {col}"):
                                    treatment_method = st.radio(
                                        f"Select treatment method for {col}:",
                                        ["Cap (Clip to bounds)", "Remove (Delete rows)", "Log Transform"],
                                        key=f"treatment_{col}"
                                    )
                                    
                                    method_map = {
                                        "Cap (Clip to bounds)": "cap",
                                        "Remove (Delete rows)": "remove",
                                        "Log Transform": "log"
                                    }
                                    
                                    if st.button(f"Apply Treatment to {col}", key=f"apply_treatment_{col}"):
                                        try:
                                            method = method_map[treatment_method]
                                            st.session_state['cleaned_df'], report = treat_outliers(
                                                st.session_state['cleaned_df'], 
                                                col, 
                                                method=method
                                            )
                                            
                                            st.success(f"✅ {report['treatment_desc']}")
                                            st.write(f"- Outliers before: {report['outliers_before']}")
                                            st.write(f"- Outliers after: {report['outliers_after']}")
                                            st.write(f"- Rows: {report['rows_before']} → {report['rows_after']}")
                                            logger.info(f"Treated outliers in {col} using {method}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error treating outliers: {e}")
                                            logger.error(f"Outlier treatment error: {e}")
                            else:
                                st.success(f"✅ No outliers detected in {col}")
                            
                            st.markdown('<div class="plot-divider"></div>', unsafe_allow_html=True)
                        
                        # Treat all outliers at once
                        st.markdown("---")
                        st.markdown("#### 🔧 Treat All Outliers")
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            global_method = st.selectbox(
                                "Select global treatment method:",
                                ["Cap (Clip to bounds)", "Remove (Delete rows)"],
                                key="global_treatment_method"
                            )
                        
                        with col2:
                            if st.button("Apply to All Columns", key="apply_all_treatments", width="stretch"):
                                try:
                                    method = "cap" if "Cap" in global_method else "remove"
                                    with st.spinner("Treating all outliers..."):
                                        st.session_state['cleaned_df'], summary_report = treat_all_outliers(
                                            st.session_state['cleaned_df'], 
                                            method=method
                                        )
                                        
                                        st.success(f"✅ Treated outliers in {summary_report['columns_treated']} columns!")
                                        st.write(f"- Total outliers before: {summary_report['total_outliers_before']}")
                                        st.write(f"- Total outliers after: {summary_report['total_outliers_after']}")
                                        st.write(f"- Dataset rows: {summary_report['rows_before']} → {summary_report['rows_after']}")
                                        logger.info(f"Treated all outliers using {method}")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error treating all outliers: {e}")
                                    logger.error(f"Treat all outliers error: {e}")
                        
                        logger.info("Outlier tab displayed successfully")
                    else:
                        st.info("No numerical columns available for outlier detection.")

                # Tab 7: Machine Learning
                with tab7:
                    st.markdown("### 🤖 Machine Learning")
                    try:
                        # Initialize session state for task_type if not set
                        if 'task_type' not in st.session_state:
                            st.session_state['task_type'] = None

                        # Task type selection
                        task_type = st.selectbox(
                            "Select Task Type", 
                            ["Select a task type", "Classification", "Regression"], 
                            key="task_type_select",
                            index=0 if st.session_state['task_type'] is None else 
                                (1 if st.session_state['task_type'] == 'classification' else 2)
                        )

                        # Update session state based on selection
                        if task_type != "Select a task type":
                            st.session_state['task_type'] = task_type.lower()
                        else:
                            st.session_state['task_type'] = None

                        if st.session_state['task_type'] is None:
                            st.warning("Please select a task type (Classification or Regression).")
                        else:
                            # Filter target columns based on the cleaned dataset
                            target_columns = (
                                st.session_state['cleaned_df'].select_dtypes(include=['object', 'category']).columns.tolist()
                                if st.session_state['task_type'] == "classification"
                                else st.session_state['cleaned_df'].select_dtypes(include=['float64', 'int64']).columns.tolist()
                            )
                            model_options = (
                                ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "DecisionTreeClassifier", "SVC"]
                                if st.session_state['task_type'] == "classification"
                                else ["LinearRegression", "RandomForestRegressor", "XGBRegressor", "DecisionTreeRegressor", "SVR"]
                            )

                            if target_columns:
                                # Target column selection
                                target_column = st.selectbox(
                                    "Select Target Column", 
                                    ["Select a target column"] + target_columns, 
                                    key="target_column_select",
                                    index=0
                                )
                                # Model selection
                                model_type = st.selectbox(
                                    "Select Model", 
                                    ["Select a model"] + model_options, 
                                    key="model_type_select",
                                    index=0
                                )
                                # Hyperparameter tuning option
                                tune_params = st.checkbox("Enable Hyperparameter Tuning (Slower)", key="tune_params")
                                # Display selected options
                                if st.session_state['task_type'] and target_column != "Select a target column" and model_type != "Select a model":
                                    st.info(f"**Selected Options**: Task Type = {st.session_state['task_type'].capitalize()}, "
                                            f"Target Column = {target_column}, Model = {model_type}, "
                                            f"Tuning = {'On' if tune_params else 'Off'}")
                                
                                # Train Model button
                                if st.button("Train Model", key="train_model", 
                                            disabled=not (st.session_state['task_type'] and 
                                                        target_column != "Select a target column" and 
                                                        model_type != "Select a model")):
                                    with st.spinner("Training model..."):
                                        try:
                                            model, report, cm, cm_fig, features, label_encoder = train_model(
                                                st.session_state['cleaned_df'], 
                                                target_column, 
                                                st.session_state['task_type'], 
                                                model_type,
                                                tune_params=tune_params
                                            )
                                            st.session_state['model'] = model
                                            st.session_state['features'] = features
                                            st.session_state['label_encoder'] = label_encoder
                                            st.session_state['ml_report'] = report
                                            st.success("Model trained successfully!")
                                            
                                            st.markdown("#### Model Evaluation")
                                            if st.session_state['task_type'] == 'classification':
                                                st.write("**Classification Report**:")
                                                st.dataframe(pd.DataFrame(report).transpose(), width="stretch")
                                                if cm is not None and cm_fig is not None:
                                                    st.markdown("#### Confusion Matrix")
                                                    st.plotly_chart(cm_fig, width="stretch")
                                                    st.download_button(
                                                        label="📥 Download Confusion Matrix",
                                                        data=download_plot(cm_fig, "confusion_matrix.html"),
                                                        file_name="confusion_matrix.html",
                                                        mime="text/html",
                                                        key="download_cm"
                                                    )
                                            else:
                                                st.write("**Regression Metrics**:")
                                                st.write(f"- Mean Squared Error: {report['Mean Squared Error']:.4f}")
                                                st.write(f"- Mean Absolute Error: {report['Mean Absolute Error']:.4f}")
                                                st.write(f"- R² Score: {report['R² Score']:.4f}")
                                                st.write(f"- Cross-Validation Score: {report['Cross_Validation_Score']:.4f}")
                                            
                                            if 'Feature_Importance' in report:
                                                st.markdown("#### Feature Importance")
                                                st.dataframe(pd.DataFrame(report['Feature_Importance']), width="stretch")
                                            
                                            # Download trained model
                                            st.markdown("---")
                                            st.markdown("#### 💾 Download Trained Model")
                                            st.download_button(
                                                label="📥 Download Model (.pkl)",
                                                data=download_model(model, f"{model_type}.pkl"),
                                                file_name=f"{model_type}_{target_column}.pkl",
                                                mime="application/octet-stream",
                                                key="download_model"
                                            )
                                            
                                            logger.info(f"Model {model_type} trained for {st.session_state['task_type']}")
                                        except Exception as e:
                                            st.error(f"Error training model: {e}")
                                            logger.error(f"Model training error: {e}")
                            else:
                                st.warning(f"No suitable columns for {st.session_state['task_type'].capitalize()}. Please check your dataset.")
                    except Exception as e:
                        st.error(f"Machine learning error: {e}")
                        logger.error(f"Machine learning error: {e}")

                # Tab 8: Enhanced Insights
                with tab8:
                    st.markdown("### 📊 Data Insights & Analysis")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("🚀 Generate Enhanced Insights", key="generate_enhanced_insights", width="stretch"):
                            with st.spinner("🔮 Analyzing your dataset..."):
                                try:
                                    # Get traditional insights first
                                    insights, recommendations = generate_insights(st.session_state['cleaned_df'])
                                    summary = get_summary(st.session_state['cleaned_df'])
                                    
                                    cleaning_report = st.session_state.get('cleaning_report', None)
                                    ml_report = st.session_state.get('ml_report', None)
                                    
                                    # Enhance with LLM
                                    enhanced_insights = enhance_insights_with_llm(
                                        insights,
                                        recommendations,
                                        summary,
                                        cleaning_report,
                                        ml_report
                                    )
                                    
                                    st.session_state['enhanced_insights'] = enhanced_insights
                                    st.success("✅ Enhanced insights generated successfully!")
                                    logger.info("Enhanced insights generated successfully")
                                except Exception as e:
                                    st.error(f"❌ Error generating enhanced insights: {e}")
                                    logger.error(f"Enhanced insights error: {e}")
                    
                    with col2:
                        if st.button("⚡ Generate Quick Summary", key="generate_quick_summary", width="stretch"):
                            with st.spinner("⚡ Generating quick summary..."):
                                try:
                                    insights, recommendations = generate_insights(st.session_state['cleaned_df'])
                                    summary = get_summary(st.session_state['cleaned_df'])
                                    
                                    quick_summary = generate_quick_summary(summary, insights, recommendations)
                                    st.session_state['enhanced_insights'] = quick_summary
                                    st.success("✅ Quick summary generated!")
                                    logger.info("Quick summary generated successfully")
                                except Exception as e:
                                    st.error(f"❌ Error generating quick summary: {e}")
                                    logger.error(f"Quick summary error: {e}")
                    
                    # Display enhanced insights if available
                    if 'enhanced_insights' in st.session_state and st.session_state['enhanced_insights']:
                        st.markdown('<div class="insight-content">', unsafe_allow_html=True)
                        st.markdown(st.session_state['enhanced_insights'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download insights as text file
                        st.download_button(
                            label="📥 Download Insights Report",
                            data=st.session_state['enhanced_insights'],
                            file_name=f"insights_report_{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.info("👆 Click 'Generate Enhanced Insights' for comprehensive analysis or 'Generate Quick Summary' for a fast overview!")
                    
                    # Show raw statistical insights as reference
                    with st.expander("📊 View Raw Statistical Data", expanded=False):
                        try:
                            insights, recommendations = generate_insights(st.session_state['cleaned_df'])
                            st.markdown("#### Statistical Observations")
                            if insights:
                                for insight in insights:
                                    st.markdown(f"- {insight}")
                            else:
                                st.info("No significant statistical observations.")
                            
                            st.markdown("#### Technical Recommendations")
                            if recommendations:
                                for recommendation in recommendations:
                                    st.markdown(f"- {recommendation}")
                            else:
                                st.info("No technical recommendations.")
                            logger.info("Raw statistical data displayed successfully")
                        except Exception as e:
                            st.error(f"Error generating statistical data: {e}")
                            logger.error(f"Statistical data error: {e}")

                # Download Cleaned Dataset
                st.markdown("### Export Cleaned Dataset")
                try:
                    st.markdown(
                        get_download_link(st.session_state['cleaned_df'], filename=f"cleaned_{uploaded_file.name}"),
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error generating download link: {e}")
                    logger.error(f"Download link error: {e}")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            logger.error(f"Dataset loading error: {e}")
    else:
        if 'last_uploaded_file' in st.session_state:
            st.session_state['last_uploaded_file'] = None
            st.session_state['df'] = None
            st.session_state['cleaned_df'] = None
            logger.info("Session state cleared.")
        st.info("Please upload a CSV or Excel file to start analyzing.")