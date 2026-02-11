import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_numerical_distribution(df, col):
    """
    Create a distribution plot for a numerical column.
    Shows histogram only (box plot shown separately in Outliers tab).
    
    Args:
        df: DataFrame
        col: Column name
    
    Returns:
        Plotly figure
    """
    try:
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name='Distribution',
                marker_color='#636EFA',
                opacity=0.7,
                nbinsx=30
            )
        )
        
        # Calculate statistics for display
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        
        # Add vertical lines for mean and median
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.2f}", 
                     annotation_position="top")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {median_val:.2f}", 
                     annotation_position="bottom")
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text=f"📊 Distribution of {col}",
            title_font_size=16,
            xaxis_title=col,
            yaxis_title="Frequency"
        )
        
        logger.info(f"Created numerical distribution for {col}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating numerical distribution for {col}: {e}")
        return None


def create_categorical_distribution(df, col):
    """
    Create a bar chart for categorical column with value counts.
    
    Args:
        df: DataFrame
        col: Column name
    
    Returns:
        Plotly figure
    """
    try:
        # Get value counts
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        
        # Calculate percentages
        value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
        
        # Limit to top 20 categories if too many
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            title_suffix = " (Top 20)"
        else:
            title_suffix = ""
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts[col],
                y=value_counts['Count'],
                text=[f"{count} ({pct}%)" for count, pct in zip(value_counts['Count'], value_counts['Percentage'])],
                textposition='auto',
                marker_color='#AB63FA',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"📊 Categorical Distribution: {col}{title_suffix}",
            xaxis_title=col,
            yaxis_title="Count",
            height=500,
            showlegend=False,
            title_font_size=16
        )
        
        # Rotate x-axis labels if many categories
        if len(value_counts) > 10:
            fig.update_xaxes(tickangle=-45)
        
        logger.info(f"Created categorical distribution for {col}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating categorical distribution for {col}: {e}")
        return None


def create_all_distributions(df):
    """
    Create individual distribution plots for all columns in the dataset.
    
    Args:
        df: DataFrame
    
    Returns:
        Dictionary with 'numerical' and 'categorical' lists of figures
    """
    try:
        figures = {'numerical': [], 'categorical': []}
        
        # Numerical columns
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig = create_numerical_distribution(df, col)
            if fig:
                figures['numerical'].append({'column': col, 'figure': fig})
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            fig = create_categorical_distribution(df, col)
            if fig:
                figures['categorical'].append({'column': col, 'figure': fig})
        
        logger.info(f"Created distributions for {len(figures['numerical'])} numerical and {len(figures['categorical'])} categorical columns")
        return figures
        
    except Exception as e:
        logger.error(f"Error creating all distributions: {e}")
        return {'numerical': [], 'categorical': []}


def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numerical columns.
    
    Args:
        df: DataFrame
    
    Returns:
        Plotly figure
    """
    try:
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(num_cols) < 2:
            logger.warning("Not enough numerical columns for correlation heatmap")
            return None
        
        corr_matrix = df[num_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='📊 Correlation Heatmap',
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(
            height=600,
            width=800,
            title_font_size=16
        )
        
        logger.info("Created correlation heatmap")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return None


def create_correlation_scatter(df):
    """
    Create scatter plot for the two most correlated numerical columns.
    
    Args:
        df: DataFrame
    
    Returns:
        Plotly figure or None
    """
    try:
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(num_cols) < 2:
            return None
        
        # Find highest correlation
        corr_matrix = df[num_cols].corr()
        corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Get the pair with highest absolute correlation
        max_corr_idx = corr_matrix.abs().stack().idxmax()
        col1, col2 = max_corr_idx
        corr_value = corr_matrix.loc[col1, col2]
        
        # Sample data if too large
        max_rows = 5000
        if df.shape[0] > max_rows:
            sample_df = df[[col1, col2]].sample(n=max_rows, random_state=42)
        else:
            sample_df = df[[col1, col2]]
        
        # Create scatter plot
        fig = px.scatter(
            sample_df,
            x=col1,
            y=col2,
            title=f'📊 Correlation Scatter: {col1} vs {col2} (r = {corr_value:.2f})',
            trendline="ols",
            opacity=0.6,
            color_discrete_sequence=['#EF553B']
        )
        
        fig.update_layout(
            height=500,
            title_font_size=16
        )
        
        logger.info(f"Created correlation scatter for {col1} vs {col2}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation scatter: {e}")
        return None


def get_outlier_info(df, col):
    """
    Get outlier information using IQR method.
    
    Args:
        df: DataFrame
        col: Column name
    
    Returns:
        Dictionary with outlier information
    """
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        return {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'total_rows': len(df)
        }
        
    except Exception as e:
        logger.error(f"Error getting outlier info for {col}: {e}")
        return None


def get_iqr_bounds(col):
    """
    Calculate IQR bounds for a column.
    
    Args:
        col: Pandas Series
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


def treat_outliers(df, column, method='cap'):
    """
    Treat outliers in a column using specified method.
    
    Args:
        df: DataFrame
        column: Column name
        method: Treatment method - 'cap' (default), 'remove', or 'log'
    
    Returns:
        Treated DataFrame and treatment report
    """
    try:
        df_treated = df.copy()
        lower_bound, upper_bound = get_iqr_bounds(df[column])
        
        # Count outliers before treatment
        outliers_before = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
        
        if method == 'cap':
            # Cap outliers at bounds
            df_treated[column] = df_treated[column].clip(lower_bound, upper_bound)
            treatment_desc = f"Capped values to [{lower_bound:.2f}, {upper_bound:.2f}]"
            
        elif method == 'remove':
            # Remove rows with outliers
            df_treated = df_treated[(df_treated[column] >= lower_bound) & (df_treated[column] <= upper_bound)]
            treatment_desc = f"Removed {len(df) - len(df_treated)} rows with outliers"
            
        elif method == 'log':
            # Log transformation (only for positive values)
            if (df[column] > 0).all():
                df_treated[column] = np.log1p(df_treated[column])
                treatment_desc = "Applied log transformation"
            else:
                raise ValueError("Log transformation requires all positive values")
        
        # Count outliers after treatment
        outliers_after = len(df_treated[(df_treated[column] < lower_bound) | (df_treated[column] > upper_bound)])
        
        report = {
            'column': column,
            'method': method,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_before': outliers_before,
            'outliers_after': outliers_after,
            'rows_before': len(df),
            'rows_after': len(df_treated),
            'treatment_desc': treatment_desc
        }
        
        logger.info(f"Treated outliers in {column} using {method} method")
        return df_treated, report
        
    except Exception as e:
        logger.error(f"Error treating outliers in {column}: {e}")
        raise


def treat_all_outliers(df, method='cap', exclude_cols=None):
    """
    Treat outliers in all numerical columns.
    
    Args:
        df: DataFrame
        method: Treatment method - 'cap', 'remove', or 'log'
        exclude_cols: List of columns to exclude from treatment
    
    Returns:
        Treated DataFrame and treatment report
    """
    try:
        df_treated = df.copy()
        reports = []
        exclude_cols = exclude_cols or []
        
        num_cols = df_treated.select_dtypes(include=['float64', 'int64']).columns
        cols_to_treat = [col for col in num_cols if col not in exclude_cols]
        
        for col in cols_to_treat:
            try:
                df_treated, report = treat_outliers(df_treated, col, method)
                reports.append(report)
            except Exception as e:
                logger.warning(f"Could not treat outliers in {col}: {e}")
                continue
        
        summary_report = {
            'method': method,
            'columns_treated': len(reports),
            'total_outliers_before': sum(r['outliers_before'] for r in reports),
            'total_outliers_after': sum(r['outliers_after'] for r in reports),
            'rows_before': len(df),
            'rows_after': len(df_treated),
            'detailed_reports': reports
        }
        
        logger.info(f"Treated outliers in {len(reports)} columns using {method} method")
        return df_treated, summary_report
        
    except Exception as e:
        logger.error(f"Error treating all outliers: {e}")
        raise