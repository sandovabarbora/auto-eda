"""
Visualization utility functions for EDA app
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Default color schemes
# OICT color palette
OICT_COLORS = {
    'purple': '#574494',
    'yellow': '#FFE14F',
    'orange': '#E37222',
    'green': '#74ECA1',
    'black': '#000000',
    'white': '#FFFFFF'
}

# Default color schemes based on OICT palette
DEFAULT_COLORS = [OICT_COLORS['purple'], OICT_COLORS['orange'], OICT_COLORS['yellow'], OICT_COLORS['green'], '#7371FC']
SEQUENTIAL_COLORSCALE = [[0, '#f0ecff'], [0.5, '#a08fd3'], [1, OICT_COLORS['purple']]]  # Custom purple scale
DIVERGING_COLORSCALE = [[0, OICT_COLORS['orange']], [0.5, '#FFFFFF'], [1, OICT_COLORS['purple']]]

def create_bar_chart(df, x_col, y_col, title=None, color_col=None):
    """
    Create a bar chart with Plotly Express
    
    Args:
        df: DataFrame with the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        title: Chart title
        color_col: Column to use for color
        
    Returns:
        Plotly figure
    """
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        color_discrete_sequence=DEFAULT_COLORS if color_col is not None else [DEFAULT_COLORS[0]]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    return fig

def create_histogram(series, bins=30, title=None, show_kde=True):
    """
    Create a histogram with optional KDE
    
    Args:
        series: Series with data to plot
        bins: Number of bins
        title: Chart title
        show_kde: Whether to show KDE curve
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        series,
        nbins=bins,
        title=title,
        opacity=0.7,
        histnorm="probability density" if show_kde else None,
        color_discrete_sequence=[DEFAULT_COLORS[0]]
    )
    
    if show_kde:
        # Add KDE plot
        kde_x = np.linspace(series.min(), series.max(), 1000)
        from scipy.stats import gaussian_kde
        kde_y = gaussian_kde(series.dropna())(kde_x)
        
        fig.add_trace(
            go.Scatter(
                x=kde_x, 
                y=kde_y, 
                mode='lines', 
                name='KDE',
                line=dict(color=DEFAULT_COLORS[1], width=2)
            )
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, title=None, color_col=None, size_col=None, add_trendline=False):
    """
    Create a scatter plot with optional trendline
    
    Args:
        df: DataFrame with the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        title: Chart title
        color_col: Column to use for color
        size_col: Column to use for point sizes
        add_trendline: Whether to add a trendline
        
    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        size=size_col,
        trendline="ols" if add_trendline else None,
        color_discrete_sequence=DEFAULT_COLORS if color_col is not None else [DEFAULT_COLORS[0]]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    return fig

def create_correlation_heatmap(corr_matrix, title=None):
    """
    Create a correlation heatmap
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale=DIVERGING_COLORSCALE,
        zmin=-1,
        zmax=1,
        title=title
    )
    
    fig.update_layout(
        height=600,
        width=800,
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

def create_box_plot(df, y_col, x_col=None, title=None, color_col=None, points='outliers'):
    """
    Create a box plot
    
    Args:
        df: DataFrame with the data
        y_col: Column to plot
        x_col: Optional column for x-axis grouping
        title: Chart title
        color_col: Column to use for color
        points: 'outliers', 'all', or False
        
    Returns:
        Plotly figure
    """
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        points=points,
        color_discrete_sequence=DEFAULT_COLORS if color_col is not None else [DEFAULT_COLORS[0]]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    return fig

def create_line_chart(df, x_col, y_col, title=None, color_col=None):
    """
    Create a line chart
    
    Args:
        df: DataFrame with the data
        x_col: Column to use for x-axis
        y_col: Column to use for y-axis
        title: Chart title
        color_col: Column to use for color
        
    Returns:
        Plotly figure
    """
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        markers=True,
        color_discrete_sequence=DEFAULT_COLORS if color_col is not None else [DEFAULT_COLORS[0]]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    return fig

def create_pie_chart(df, names_col, values_col, title=None):
    """
    Create a pie chart
    
    Args:
        df: DataFrame with the data
        names_col: Column to use for slice names
        values_col: Column to use for slice values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title,
        color_discrete_sequence=DEFAULT_COLORS,
        hole=0.4
    )
    
    fig.update_traces(textinfo='percent+label')
    
    return fig