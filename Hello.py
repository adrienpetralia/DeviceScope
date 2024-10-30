"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st
from st_pages import Page, show_pages, add_page_title

from utils import *

# Specify what pages should be shown in the sidebar
show_pages(
	[
		Page("Hello.py", "DeviceScope", ":zap:"),  # Home emoji is correct
		Page("Pages/Playground.py", "Playground", ":mag:"),  # Changed from :books: to a book emoji
		Page("pages/Benchmark.py", "Benchmark", ":mag:"), 
		Page("pages/WhatsBehind.py", "What's behind", ":mag:"), 
	]
)

add_page_title()

st.write("# DeviceScope")

st.markdown(
    """
    Welcome to DeviceScope! :mag: :zap: This app provides an analytical tool to browse, detect and localize appliance patterns in electricity consumption time series.
    """
)

st.markdown(f"""
            1. **Explore** electrical consumption series at different sampling rates.
            2. **Detect** appliances in a period of time using the different trained classifiers and compare their performance.
            3. **Localize** appliance patterns using explainable classification approaches (CAM/AttMap).
            """)

