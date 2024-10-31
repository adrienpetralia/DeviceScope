"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st
from constants import *

from utils import *
from constants import *


st.markdown(text_tab_benchmark)

tab_playground, tab_benchmark = st.tabs(
        ["Metrics Comparaison", "Visual Comparaison"]
    )
    
with tab_playground:
    run_metric_frame()

with tab_benchmark:
    run_nilmmodel_frame()

