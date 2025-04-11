import streamlit as st
import pandas as pd
import json
import requests
from eda import EDA
from utils.ui_components import create_section_header, create_card, OICT_COLORS

def api_loader_component():
    """Component for loading data from an API endpoint"""
    
    create_section_header("API Data Loader", "Connect to REST APIs to fetch data", "🔌")
    
    # Create a card for API connection
    st.markdown("""
    <div style="
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        border-top: 3px solid #574494;
    ">
    """, unsafe_allow_html=True)
    
    # API URL input
    api_url = st.text_input(
        "API URL",
        placeholder="https://api.example.com/data",
        help="Enter the full URL of the API endpoint"
    )
    
    # Setup columns for method & auth
    col1, col2 = st.columns(2)
    
    with col1:
        # HTTP Method
        api_method = st.selectbox(
            "HTTP Method",
            options=["GET", "POST"],
            index=0,
            help="Select the HTTP method to use for the API request"
        )
    
    with col2:
        # Authentication Type
        auth_type = st.selectbox(
            "Authentication",
            options=["None", "API Key", "Bearer Token", "Basic Auth"],
            index=0,
            help="Select the authentication method required by the API"
        )
    
    # Authentication details based on selected type
    if auth_type == "API Key":
        col1, col2 = st.columns(2)
        with col1:
            api_key_name = st.text_input(
                "API Key Name",
                value="api-key",
                help="The name of the API key parameter (e.g., 'api-key', 'apikey', 'key')"
            )
        with col2:
            api_key_value = st.text_input(
                "API Key Value",
                type="password",
                help="Your API key"
            )
            
        api_key_location = st.radio(
            "API Key Location",
            options=["Header", "Query Parameter"],
            horizontal=True,
            help="Where to include the API key in the request"
        )
        
    elif auth_type == "Bearer Token":
        bearer_token = st.text_input(
            "Bearer Token",
            type="password",
            help="Your Bearer token (without 'Bearer' prefix)"
        )
        
    elif auth_type == "Basic Auth":
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input(
                "Username",
                help="Username for Basic Authentication"
            )
        with col2:
            password = st.text_input(
                "Password",
                type="password",
                help="Password for Basic Authentication"
            )
    
    # Additional request configuration (headers, body, etc.)
    with st.expander("Advanced Settings", expanded=False):
        # Custom headers
        st.subheader("HTTP Headers")
        add_headers = st.checkbox(
            "Add Custom Headers",
            value=False,
            help="Add custom HTTP headers to the request"
        )
        
        headers = {}
        if add_headers:
            header_count = st.number_input(
                "Number of Headers",
                min_value=1,
                max_value=10,
                value=1,
                help="Number of custom headers to add"
            )
            
            for i in range(int(header_count)):
                col1, col2 = st.columns(2)
                with col1:
                    header_name = st.text_input(f"Header Name #{i+1}", key=f"header_name_{i}")
                with col2:
                    header_value = st.text_input(f"Header Value #{i+1}", key=f"header_value_{i}")
                
                if header_name:
                    headers[header_name] = header_value
        
        # Request body for POST
        if api_method == "POST":
            st.subheader("Request Body")
            body_type = st.radio(
                "Body Type",
                options=["JSON", "Form Data"],
                horizontal=True,
                help="Format of the request body"
            )
            
            if body_type == "JSON":
                request_body = st.text_area(
                    "JSON Body",
                    value="{}",
                    height=150,
                    help="Enter JSON request body"
                )
                
                # Validate JSON
                try:
                    json.loads(request_body)
                    st.success("Valid JSON ✓")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {str(e)}")
            else:  # Form Data
                st.write("Form Data Parameters:")
                form_count = st.number_input(
                    "Number of Parameters",
                    min_value=1,
                    max_value=10,
                    value=1
                )
                
                form_data = {}
                for i in range(int(form_count)):
                    col1, col2 = st.columns(2)
                    with col1:
                        param_name = st.text_input(f"Parameter Name #{i+1}", key=f"param_name_{i}")
                    with col2:
                        param_value = st.text_input(f"Parameter Value #{i+1}", key=f"param_value_{i}")
                    
                    if param_name:
                        form_data[param_name] = param_value
        
        # Response handling options
        st.subheader("Response Handling")
        json_path = st.text_input(
            "JSON Path to Data",
            placeholder="data.results",
            help="Path to the data array in the JSON response (e.g., 'data.items', 'results')"
        )
        
        flatten_nested = st.checkbox(
            "Flatten Nested Objects",
            value=True,
            help="Convert nested JSON structures into flattened column names"
        )
        
        normalize_arrays = st.checkbox(
            "Normalize Arrays",
            value=True,
            help="Convert JSON arrays to separate rows"
        )
    
    # Send request button
    if st.button("Fetch Data from API", type="primary", use_container_width=True):
        if not api_url:
            st.error("Please enter an API URL")
        else:
            with st.spinner("Sending API request..."):
                try:
                    # Prepare request
                    request_kwargs = {"headers": {}}
                    
                    # Add authentication
                    if auth_type == "API Key":
                        if api_key_location == "Header":
                            request_kwargs["headers"][api_key_name] = api_key_value
                        else:  # Query Parameter
                            if "?" in api_url:
                                api_url += f"&{api_key_name}={api_key_value}"
                            else:
                                api_url += f"?{api_key_name}={api_key_value}"
                    
                    elif auth_type == "Bearer Token":
                        request_kwargs["headers"]["Authorization"] = f"Bearer {bearer_token}"
                    
                    elif auth_type == "Basic Auth":
                        request_kwargs["auth"] = (username, password)
                    
                    # Add custom headers
                    if headers:
                        request_kwargs["headers"].update(headers)
                    
                    # Add request body for POST
                    if api_method == "POST":
                        if body_type == "JSON":
                            request_kwargs["json"] = json.loads(request_body)
                        else:  # Form Data
                            request_kwargs["data"] = form_data
                    
                    # Make the request
                    if api_method == "GET":
                        response = requests.get(api_url, **request_kwargs)
                    else:  # POST
                        response = requests.post(api_url, **request_kwargs)
                    
                    # Check status
                    response.raise_for_status()
                    
                    # Parse response
                    response_json = response.json()
                    
                    # Navigate to nested data if needed
                    if json_path:
                        try:
                            path_parts = json_path.split('.')
                            data_obj = response_json
                            for part in path_parts:
                                data_obj = data_obj[part]
                            response_json = data_obj
                        except (KeyError, TypeError):
                            st.error(f"Could not find data at JSON path: '{json_path}'")
                            st.json(response_json)
                            st.markdown("</div>", unsafe_allow_html=True)
                            return
                    
                    # Convert to DataFrame
                    if isinstance(response_json, list):
                        df = pd.json_normalize(response_json) if flatten_nested else pd.DataFrame(response_json)
                    elif isinstance(response_json, dict):
                        if normalize_arrays:
                            # Find list fields to normalize
                            array_fields = [k for k, v in response_json.items() if isinstance(v, list)]
                            if array_fields and len(array_fields) == 1:
                                # If there's a single array field, normalize it
                                df = pd.json_normalize(response_json[array_fields[0]]) if flatten_nested else pd.DataFrame(response_json[array_fields[0]])
                            else:
                                # Otherwise treat as a single record
                                df = pd.json_normalize([response_json]) if flatten_nested else pd.DataFrame([response_json])
                        else:
                            df = pd.json_normalize([response_json]) if flatten_nested else pd.DataFrame([response_json])
                    else:
                        st.error("API response is not in a valid JSON format (list or object)")
                        st.write(response_json)
                        st.markdown("</div>", unsafe_allow_html=True)
                        return
                    
                    # Store in session state
                    st.session_state.data = df
                    st.session_state.eda = EDA(df)
                    
                    # Display success message
                    st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns from API")
                    
                    # Preview data
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                
                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Error: {str(e)}")
                
                except ValueError as e:
                    st.error(f"JSON Parsing Error: {str(e)}")
                
                except Exception as e:
                    st.error(f"Unexpected Error: {str(e)}")
    
    # Close container div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sample APIs section
    create_section_header("Sample APIs", "Try these public APIs to test the loader", "🌐")
    
    # Sample API cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            height: 100%;
            border-top: 3px solid #E37222;
        ">
            <h4 style="color: #574494; margin-bottom: 1rem;">NASA APOD API</h4>
            <p style="margin-bottom: 1rem;">NASA Astronomy Picture of the Day API provides beautiful astronomy images.</p>
            <p><strong>URL:</strong> https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY&count=10</p>
            <p><strong>Method:</strong> GET</p>
            <p><strong>Auth:</strong> None (uses demo key in URL)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            height: 100%;
            border-top: 3px solid #74ECA1;
        ">
            <h4 style="color: #574494; margin-bottom: 1rem;">JSON Placeholder API</h4>
            <p style="margin-bottom: 1rem;">Free fake API for testing with various endpoints.</p>
            <p><strong>URL:</strong> https://jsonplaceholder.typicode.com/users</p>
            <p><strong>Method:</strong> GET</p>
            <p><strong>Auth:</strong> None</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional API examples
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            height: 100%;
            border-top: 3px solid #FFE14F;
        ">
            <h4 style="color: #574494; margin-bottom: 1rem;">Open Weather Map</h4>
            <p style="margin-bottom: 1rem;">Current weather data for cities worldwide.</p>
            <p><strong>URL:</strong> https://samples.openweathermap.org/data/2.5/weather?q=London,uk&appid=b6907d289e10d714a6e88b30761fae22</p>
            <p><strong>Method:</strong> GET</p>
            <p><strong>Auth:</strong> None (sample API key included)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            height: 100%;
            border-top: 3px solid #574494;
        ">
            <h4 style="color: #574494; margin-bottom: 1rem;">Random User Generator</h4>
            <p style="margin-bottom: 1rem;">API for generating random user data.</p>
            <p><strong>URL:</strong> https://randomuser.me/api/?results=100</p>
            <p><strong>Method:</strong> GET</p>
            <p><strong>Auth:</strong> None</p>
            <p><strong>JSON Path:</strong> results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Help section
    create_card(
        content="""
        <h4 style="margin-top: 0;">Tips for API Data Loading</h4>
        <ul>
            <li><strong>Authentication:</strong> Ensure you have the correct API keys or tokens for authenticated endpoints.</li>
            <li><strong>JSON Path:</strong> Use the JSON path field to extract data from nested responses (e.g., <code>data.items</code> or <code>results</code>).</li>
            <li><strong>Pagination:</strong> Some APIs return paginated results. You may need to make multiple requests to get all data.</li>
            <li><strong>Rate Limits:</strong> Be aware that many APIs have rate limits on how many requests you can make.</li>
            <li><strong>Data Transformation:</strong> You may need to transform or clean the data after importing it.</li>
        </ul>
        """,
        title="API Loading Help",
        is_info=True
    )