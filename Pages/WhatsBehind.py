"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st

from utils import *
from constants import *

st.markdown(text_about)

st.image("Figures/DeviceScopePipelineGithub.png", caption="Proposed appliance detection pipeline.")

with st.expander(f"""### Appliance detection as a time series classification problem"""):
    st.markdown(text_description_model)

with st.expander(f"""### Explainable classification to localize appliance patterns"""):
    st.write(text_description_explainability)

st.markdown("""### Smart meters datasets""")
st.markdown(text_description_dataset)

