"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st
from st_pages import Page, show_pages, add_page_title

from Utils.utils import *

# Specify what pages should be shown in the sidebar
show_pages(
	[
		Page("Hello.py", "DeviceScope", ":zap:"),  # Home emoji is correct
		Page("Pages/Playground.py", "Playground", ":control_knobs:"),  # Changed from :books: to a book emoji
		Page("Pages/Benchmark.py", "Benchmark", ":bar_chart:"), 
		Page("Pages/WhatsBehind.py", "What's behind", ":mag:"), 
	]
)

st.write("# :zap: DeviceScope")

st.markdown(
    """
    Welcome to DeviceScope! :mag: :zap: This app provides an analytical tool to browse, detect and localize appliance patterns in electricity consumption time series.
    """
)

st.markdown(f"""
            1. **Explore** electricity consumption series.
            2. **Detect** and **Localize** appliances use in a period of time.
            3. **Compare** the performance of our approach against other methods.
            """)

st.image("Figures/homepage_figure.png")


st.markdown(f"""We propose DeviceScope as an interactive app to facilitate the visualization of smart meter consumption data using appliance detection localization methods. 
            DeviceScope is based on **CamAL** *(**C**lass **A**ctivation **M**ap based **A**ppliance **L**ocalization)*, a novel weakly supervised method for appliance pattern localization which only requires the appliance's possession label for training.
            """)

st.markdown(text_info)

