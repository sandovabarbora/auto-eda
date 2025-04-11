import streamlit as st
import pandas as pd
import io
from eda import EDA
from components.api_loader import api_loader_component
from utils.ui_components import create_card, create_section_header, OICT_COLORS

def load_data_component():
    """Component for loading data into the app"""
    create_section_header("Data Loading", "Upload your data file or use one of our sample datasets", "📤")
    
    # Create tabs for different data loading methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🌐 API Connection", "💾 Sample Datasets"])
    
    with tab1:
        # Style the file uploader
        st.markdown("""
        <div style="
            background-color: white; 
            border-radius: 10px; 
            padding: 1.5rem; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
            margin-bottom: 1.5rem;
            text-align: center;
        ">
            <div style="font-size: 1.2rem; font-weight: 600; color: #574494; margin-bottom: 1rem;">
                Upload Your Dataset
            </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls", "txt"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Determine file type
            is_csv = uploaded_file.name.endswith(('.csv', '.txt'))
            
            try:
                # UI for import settings
                with st.expander("Import Settings", expanded=True):
                    if is_csv:
                        # Enhanced CSV separator selection
                        st.markdown("""
                        <div style="margin-bottom: 0.5rem; font-weight: 500; color: #574494;">
                            CSV Separator
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Common separators
                        separator_options = {
                            "Comma (,)": ",",
                            "Semicolon (;)": ";",
                            "Tab (\\t)": "\t",
                            "Pipe (|)": "|",
                            "Space ( )": " ",
                            "Custom": "custom"
                        }
                        
                        # Create a row of buttons for separator selection
                        cols = st.columns(len(separator_options))
                        
                        # Initialize separator in session state if not present
                        if "csv_separator" not in st.session_state:
                            st.session_state.csv_separator = ","
                            st.session_state.separator_option = "Comma (,)"
                        
                        # Display separator buttons
                        for i, (option_name, sep_value) in enumerate(separator_options.items()):
                            with cols[i]:
                                # Check if this is the current selection
                                is_selected = st.session_state.separator_option == option_name
                                
                                # Create button with conditional styling
                                if st.button(
                                    option_name,
                                    key=f"sep_{option_name}",
                                    use_container_width=True,
                                    type="primary" if is_selected else "secondary"
                                ):
                                    st.session_state.separator_option = option_name
                                    if sep_value != "custom":
                                        st.session_state.csv_separator = sep_value
                                    st.rerun()
                        
                        # Show custom separator input if "Custom" is selected
                        if st.session_state.separator_option == "Custom":
                            custom_sep = st.text_input(
                                "Enter custom separator",
                                value=st.session_state.csv_separator if st.session_state.csv_separator not in [",", ";", "\t", "|", " "] else ""
                            )
                            if custom_sep:
                                st.session_state.csv_separator = custom_sep
                        
                        # Show the currently selected separator with a preview
                        sep_display = st.session_state.csv_separator.replace("\t", "\\t")
                        st.markdown(f"""
                        <div style="
                            margin-top: 0.75rem;
                            padding: 0.5rem;
                            background-color: #f0ecff;
                            border-radius: 5px;
                            font-family: monospace;
                        ">
                            Selected separator: <strong>'{sep_display}'</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional CSV settings
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            encoding = st.selectbox(
                                "Encoding",
                                options=["utf-8", "iso-8859-1", "windows-1250", "latin1", "utf-16"],
                                index=0
                            )
                        
                        with col2:
                            header_row = st.selectbox(
                                "Header row",
                                options=["First row", "No header"],
                                index=0
                            )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            decimal = st.text_input("Decimal Point", value=".")
                        
                        with col2:
                            thousands = st.text_input("Thousands Separator", value="")
                        
                        # Preview first few lines with current separator
                        st.markdown("""
                        <div style="margin-top: 1rem; font-weight: 500; color: #574494;">
                            File Preview with Current Separator
                        </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            # Read just the first few lines for preview
                            preview_data = pd.read_csv(
                                uploaded_file,
                                sep=st.session_state.csv_separator,
                                encoding=encoding,
                                nrows=5,
                                header=0 if header_row == "First row" else None
                            )
                            
                            # Display preview
                            st.dataframe(preview_data, use_container_width=True)
                            
                            # Show info about detected columns
                            st.info(f"Detected {len(preview_data.columns)} columns with current separator.")
                            
                            # Reset file position for later reading
                            uploaded_file.seek(0)
                        except Exception as e:
                            st.error(f"Error previewing file with current separator: {str(e)}")
                    else:  # Excel
                        sheet_name = st.text_input("Sheet name (leave empty for first sheet)", value="")
                        header_row = st.selectbox(
                            "Header row",
                            options=["First row", "No header"],
                            index=0
                        )
                
                # Load button with prominent styling
                st.markdown("""
                <div style="margin-top: 1.5rem; margin-bottom: 1rem; text-align: center;">
                """, unsafe_allow_html=True)
                
                if st.button("Load Data", type="primary", use_container_width=True):
                    with st.spinner("Loading data..."):
                        # Load the data with specified settings
                        if is_csv:
                            data = pd.read_csv(
                                uploaded_file, 
                                sep=st.session_state.csv_separator,
                                encoding=encoding,
                                header=0 if header_row == "First row" else None,
                                decimal=decimal,
                                thousands=thousands if thousands else None
                            )
                            
                            # If no header, create column names
                            if header_row == "No header":
                                data.columns = [f"Column_{i+1}" for i in range(len(data.columns))]
                        else:  # Excel
                            # Load the data with specified settings
                            if sheet_name.strip():
                                data = pd.read_excel(
                                    uploaded_file, 
                                    sheet_name=sheet_name,
                                    header=0 if header_row == "First row" else None
                                )
                            else:
                                data = pd.read_excel(
                                    uploaded_file,
                                    header=0 if header_row == "First row" else None
                                )
                            
                            # If no header, create column names
                            if header_row == "No header":
                                data.columns = [f"Column_{i+1}" for i in range(len(data.columns))]
                        
                        # Store in session state
                        st.session_state.data = data
                        st.session_state.eda = EDA(data)
                        
                        # Success message
                        st.success(f"✅ Data loaded successfully: {data.shape[0]:,} rows and {data.shape[1]} columns")
                        
                        # Show button to navigate to overview
                        if st.button("Go to Data Overview →", use_container_width=True):
                            st.session_state['current_page'] = "Overview"
                            st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some helpful tips
        with st.expander("CSV File Tips", expanded=False):
            st.markdown("""
            ### Common CSV Separators
            - **Comma (,)**: Standard CSV format (comma-separated values)
            - **Semicolon (;)**: Common in European countries where comma is used as decimal separator
            - **Tab (\\t)**: Tab-separated values (TSV files)
            - **Pipe (|)**: Used when data may contain commas and semicolons
            
            ### Common Issues & Solutions
            - **Encoding Problems**: If you see strange characters, try different encodings
            - **Wrong Separator**: If data loads into a single column, try a different separator
            - **Decimal Separator**: In some countries, comma is used instead of period for decimals
            - **Quoted Fields**: CSV files sometimes use quotes around fields containing separators
            """)

    with tab2:
        # API Connection tab
        api_loader_component()
    
    with tab3:
        st.markdown("""
        <div style="
            background-color: white; 
            border-radius: 10px; 
            padding: 1.5rem; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
            margin-bottom: 1.5rem;
        ">
            <div style="font-size: 1.2rem; font-weight: 600; color: #574494; margin-bottom: 1rem;">
                Sample Datasets
            </div>
            <p style="color: #666; margin-bottom: 1.5rem;">
                Choose from one of our pre-loaded datasets to explore the tool's capabilities.
            </p>
        """, unsafe_allow_html=True)
        
        # Create a grid of sample datasets with enhanced styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="
                background-color: rgba(87, 68, 148, 0.05);
                border-radius: 8px;
                padding: 1.2rem;
                height: 100%;
                border-left: 3px solid #574494;
            ">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">Iris Dataset 🌸</div>
                <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.8rem;">
                    Classic dataset for classification with flower measurements.
                </p>
                <div style="font-size: 0.8rem; margin-bottom: 1rem;">
                    <span style="font-weight: 500;">150 rows</span> • 
                    <span style="font-weight: 500;">5 columns</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load Iris Dataset", key="load_iris", use_container_width=True):
                try:
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    data = pd.DataFrame(iris.data, columns=iris.feature_names)
                    data['target'] = iris.target
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Iris dataset loaded successfully!")
                    
                    # Show button to navigate to overview
                    if st.button("Go to Data Overview →", key="goto_overview_iris", use_container_width=True):
                        st.session_state['current_page'] = "Overview"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")
        
        with col2:
            st.markdown("""
            <div style="
                background-color: rgba(231, 114, 34, 0.05);
                border-radius: 8px;
                padding: 1.2rem;
                height: 100%;
                border-left: 3px solid #E37222;
            ">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">Titanic Dataset 🚢</div>
                <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.8rem;">
                    Survival data from the Titanic shipwreck.
                </p>
                <div style="font-size: 0.8rem; margin-bottom: 1rem;">
                    <span style="font-weight: 500;">891 rows</span> • 
                    <span style="font-weight: 500;">12 columns</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load Titanic Dataset", key="load_titanic", use_container_width=True):
                try:
                    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
                    data = pd.read_csv(url)
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Titanic dataset loaded successfully!")
                    
                    # Show button to navigate to overview
                    if st.button("Go to Data Overview →", key="goto_overview_titanic", use_container_width=True):
                        st.session_state['current_page'] = "Overview"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")
        
        # Add more sample datasets
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="
                background-color: rgba(116, 236, 161, 0.05);
                border-radius: 8px;
                padding: 1.2rem;
                height: 100%;
                border-left: 3px solid #74ECA1;
                margin-top: 1rem;
            ">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">Boston Housing 🏠</div>
                <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.8rem;">
                    Housing price dataset with property attributes.
                </p>
                <div style="font-size: 0.8rem; margin-bottom: 1rem;">
                    <span style="font-weight: 500;">506 rows</span> • 
                    <span style="font-weight: 500;">14 columns</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load Boston Dataset", key="load_boston", use_container_width=True):
                try:
                    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
                    data = pd.read_csv(url)
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Boston Housing dataset loaded successfully!")
                    
                    # Show button to navigate to overview
                    if st.button("Go to Data Overview →", key="goto_overview_boston", use_container_width=True):
                        st.session_state['current_page'] = "Overview"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")
        
        with col2:
            st.markdown("""
            <div style="
                background-color: rgba(255, 225, 79, 0.05);
                border-radius: 8px;
                padding: 1.2rem;
                height: 100%;
                border-left: 3px solid #FFE14F;
                margin-top: 1rem;
            ">
                <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">Diabetes Dataset 🩺</div>
                <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.8rem;">
                    Medical dataset for diabetes prediction.
                </p>
                <div style="font-size: 0.8rem; margin-bottom: 1rem;">
                    <span style="font-weight: 500;">768 rows</span> • 
                    <span style="font-weight: 500;">9 columns</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load Diabetes Dataset", key="load_diabetes", use_container_width=True):
                try:
                    from sklearn.datasets import load_diabetes
                    diabetes = load_diabetes()
                    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                    data['target'] = diabetes.target
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Diabetes dataset loaded successfully!")
                    
                    # Show button to navigate to overview
                    if st.button("Go to Data Overview →", key="goto_overview_diabetes", use_container_width=True):
                        st.session_state['current_page'] = "Overview"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample dataset: {str(e)}")
        
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data type conversion section
    if st.session_state.data is not None:
        create_section_header("Data Type Conversion", "Convert column types if needed", "🔄")
        
        # Create an attractive card for type conversion
        st.markdown("""
        <div style="
            background-color: white; 
            border-radius: 10px; 
            padding: 1.5rem; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
            margin-bottom: 1.5rem;
        ">
        """, unsafe_allow_html=True)
        
        data = st.session_state.data
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            column_to_convert = st.selectbox("Select column", data.columns, key="col_to_convert")
            
            # Show current type
            current_type = data[column_to_convert].dtype
            st.markdown(f"""
            <div style="
                margin-top: 0.5rem;
                font-size: 0.9rem;
                color: #666;
            ">
                Current type: <span style="font-family: monospace; color: #574494; font-weight: 500;">{current_type}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            target_type = st.selectbox(
                "Convert to type",
                ["numeric", "categorical", "datetime", "boolean"],
                key="target_type"
            )
            
            # Show example conversion
            example_val = data[column_to_convert].iloc[0] if len(data) > 0 else ""
            st.markdown(f"""
            <div style="
                margin-top: 0.5rem;
                font-size: 0.9rem;
                color: #666;
            ">
                Example value: <span style="font-family: monospace;">{example_val}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.write("")  # Create some space
            st.write("")  # Create some more space
            
            if st.button("Convert", type="primary", use_container_width=True, key="convert_button"):
                try:
                    # Make a copy of the data
                    converted_data = data.copy()
                    
                    # Convert based on target type
                    if target_type == "numeric":
                        converted_data[column_to_convert] = pd.to_numeric(converted_data[column_to_convert], errors='coerce')
                        success_msg = f"Converted '{column_to_convert}' to numeric type"
                    elif target_type == "datetime":
                        converted_data[column_to_convert] = pd.to_datetime(converted_data[column_to_convert], errors='coerce')
                        success_msg = f"Converted '{column_to_convert}' to datetime type"
                    elif target_type == "boolean":
                        converted_data[column_to_convert] = converted_data[column_to_convert].astype(bool)
                        success_msg = f"Converted '{column_to_convert}' to boolean type"
                    else:  # categorical
                        converted_data[column_to_convert] = converted_data[column_to_convert].astype(str)
                        success_msg = f"Converted '{column_to_convert}' to categorical (string) type"
                    
                    # Update session state
                    st.session_state.data = converted_data
                    st.session_state.eda = EDA(converted_data)
                    
                    st.success(success_msg)
                except Exception as e:
                    st.error(f"Error converting column type: {str(e)}")
        
        # Preview of column values
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        if column_to_convert:
            values_preview = data[column_to_convert].head(5).tolist()
            values_str = ", ".join([str(v) for v in values_preview])
            
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa; 
                padding: 0.8rem; 
                border-radius: 8px; 
                font-size: 0.9rem;
                margin-top: 0.5rem;
            ">
                <span style="font-weight: 500;">First 5 values:</span> 
                <span style="font-family: monospace;">{values_str}...</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Show unique values for categorical columns
            if data[column_to_convert].dtype == 'object' or len(data[column_to_convert].unique()) < 20:
                unique_vals = data[column_to_convert].nunique()
                st.markdown(f"""
                <div style="
                    margin-top: 0.5rem;
                    font-size: 0.9rem;
                    color: #666;
                ">
                    Unique values: <span style="font-weight: 500;">{unique_vals}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)
