"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st
from constants import *

from utils import *


tab_playground, tab_benchmark, tab_about = st.tabs(
        ["Playground", "Benchmark", "About"]
    )
    
with tab_playground:
    run_metric_frame()

with tab_benchmark:
    run_nilmmodel_frame()

