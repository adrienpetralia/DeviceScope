"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st

from Utils.utils import *
from Utils.constants import *


st.markdown(text_intro_whats_behind)


tab_method_description, tab_sota_methods, tab_dataset_description = st.tabs(
    ["DeviceScope core: CamAL", "SotA NILM methods for appliance localization", "Datasets description"]
    )


with tab_method_description:

    st.markdown(text_camal_info)

    st.image("Figures/DeviceScopePipelineGithub.png", caption="Proposed appliance detection pipeline.")

    with st.expander(f"""### Explainable classification to localize appliance patterns"""):
        st.write(text_description_explainability)


with tab_sota_methods:
    st.markdown(text_description_model)


with tab_dataset_description:
    st.markdown("""### Smart meters datasets""")
    st.markdown(text_description_dataset)

