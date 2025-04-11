"""
UI component utilities for enhanced visual display
"""
import streamlit as st

# OICT color palette
OICT_COLORS = {
    'purple': '#574494',
    'yellow': '#FFE14F',
    'orange': '#E37222',
    'green': '#74ECA1',
    'black': '#000000',
    'white': '#FFFFFF'
}


def create_metric_card(title, value, description=None, icon=None, color=OICT_COLORS['purple'], delta=None):
    """
    Create a beautifully styled metric card
    
    Args:
        title: Metric title
        value: Metric value
        description: Optional description text
        icon: Optional icon emoji
        color: Card accent color
        delta: Optional delta value for showing change
    """
    # Use Streamlit columns for layout instead of custom HTML/CSS
    col = st.container()
    with col:
        # Create a cleaner card with simpler HTML
        st.markdown(f"""
        <div style="
            background-color: white; 
            border-radius: 10px; 
            padding: 1.25rem; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
            border-top: 3px solid {color}; 
            height: 100%;
            margin-bottom: 1rem;
        ">
            {f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ''}
            <div style="font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #333; margin-top: 0.25rem;">{value}</div>
            {f'<div style="font-size: 0.8rem; color: #666; margin-top: 0.25rem;">{description}</div>' if description else ''}
            {f'<div style="color: {"#74ECA1" if delta >= 0 else "#E37222"}; font-size: 0.9rem; margin-top: 0.25rem;">{("↑ " + str(abs(delta)) + "%") if delta >= 0 else ("↓ " + str(abs(delta)) + "%")}</div>' if delta is not None else ''}
        </div>
        """, unsafe_allow_html=True)

def create_metrics_row(metrics_data):
    """
    Create a row of metric cards
    
    Args:
        metrics_data: List of dicts with keys:
            - title: Metric title
            - value: Metric value
            - description (optional): Description text
            - icon (optional): Icon emoji
            - color (optional): Card accent color
            - delta (optional): Delta value for showing change
    """
    # Create columns based on number of metrics
    cols = st.columns(len(metrics_data))
    
    # Create a metric card in each column
    for i, metric in enumerate(metrics_data):
        with cols[i]:
            create_metric_card(
                title=metric['title'],
                value=metric['value'],
                description=metric.get('description'),
                icon=metric.get('icon'),
                color=metric.get('color', OICT_COLORS['purple']),
                delta=metric.get('delta')
            )

def create_section_header(title, description=None, icon=None):
    """
    Create a styled section header
    
    Args:
        title: Section title
        description: Optional section description
        icon: Optional icon emoji
    """
    icon_html = f"{icon} " if icon else ""
    description_html = f"""
    <p style="color: #666; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 1.5rem;">
        {description}
    </p>
    """ if description else ""
    
    st.markdown(f"""
    <div style="margin-top: 1.5rem; margin-bottom: 1.5rem;">
        <h2 style="color: #574494; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center;">
            {icon_html}{title}
        </h2>
        {description_html}
    </div>
    """, unsafe_allow_html=True)

def create_card(content, title=None, color=OICT_COLORS['purple'], is_success=False, is_warning=False, is_info=False):
    """
    Create a styled card with content
    
    Args:
        content: Card content (can include markdown)
        title: Optional card title
        color: Card accent color
        is_success: Whether this is a success card
        is_warning: Whether this is a warning card
        is_info: Whether this is an info card
    """
    if is_success:
        color = OICT_COLORS['green']
    elif is_warning:
        color = OICT_COLORS['orange']
    elif is_info:
        color = OICT_COLORS['yellow']
    
    title_html = f"""
    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.8rem; color: #333;">
        {title}
    </div>
    """ if title else ""
    
    st.markdown(f"""
    <div style="
        background-color: white; 
        border-radius: 10px; 
        padding: 1.25rem; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
        border-left: 4px solid {color}; 
        margin-bottom: 1.5rem;
    ">
        {title_html}
        <div>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_insight_cards(insights, color=OICT_COLORS['purple']):
    """
    Create styled insight cards for a list of insights
    
    Args:
        insights: List of insight strings
        color: Card accent color
    """
    for i, insight in enumerate(insights):
        st.markdown(f"""
        <div style="
            background-color: rgba(87, 68, 148, 0.05);
            border-left: 4px solid {color};
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin-bottom: 0.8rem;
            transition: all 0.3s;
        " onmouseover="this.style.transform='translateX(3px)';this.style.backgroundColor='rgba(87, 68, 148, 0.08)';" 
          onmouseout="this.style.transform='translateX(0px)';this.style.backgroundColor='rgba(87, 68, 148, 0.05)';">
            {insight}
        </div>
        """, unsafe_allow_html=True)

def create_tabs(tab_items, key_prefix="custom_tab"):
    """
    Create custom styled tabs
    
    Args:
        tab_items: List of tab names
        key_prefix: Prefix for session state keys
        
    Returns:
        Selected tab name
    """
    # Generate a unique key for this tab navigation
    state_key = f"{key_prefix}_selected"
    
    # Initialize the selected tab if not present
    if state_key not in st.session_state:
        st.session_state[state_key] = tab_items[0]
    
    # Create the tabs container
    st.markdown("""
    <div style="
        background-color: #f5f5fa;
        padding: 5px;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        gap: 5px;
    ">
    """, unsafe_allow_html=True)
    
    # Create tab-style buttons
    cols = st.columns(len(tab_items))
    
    for i, tab_name in enumerate(tab_items):
        # Determine if this tab is selected
        is_selected = st.session_state[state_key] == tab_name
        
        # Create the button
        if cols[i].button(
            tab_name, 
            key=f"{key_prefix}_{i}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            st.session_state[state_key] = tab_name
            st.rerun()
    
    # Close tabs container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Return the selected tab
    return st.session_state[state_key]