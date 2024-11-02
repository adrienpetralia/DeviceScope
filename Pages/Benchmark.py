"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st

from Utils.utils import *
from Utils.constants import *


st.markdown(text_tab_benchmark)

tab_playground, tab_benchmark = st.tabs(
        ["Metrics Comparaison", "Visual Comparaison"]
    )
    
with tab_playground:
    run_metric_comparaison_frame()

with tab_benchmark:
    run_visualnilmmodel_comparaison_frame()

