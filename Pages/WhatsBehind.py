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
    ["CamAL", "NILM baselines description", "Datasets description"]
    )


with tab_method_description:

    st.markdown(text_camal_info1)

    st.image("Figures/DeviceScopePipelineGithub.png", caption="Illustration of our CamAL pipeline.")

    st.markdown(text_camal_info2)

    with st.expander(f"""### ResNets Ensemble"""):
        st.write(text_description_resnets)

    st.markdown(f"""#### Step2: AppliancePattern Localization""")
    st.markdown(text_camal_info3)

    with st.expander(f"""### Class Activation Map"""):
        st.write(text_description_explainability)



with tab_sota_methods:
    st.markdown("""### Appliance localization using NILM baselines""")
    st.markdown(text_description_model)

    with st.expander(f"""### TPNILM (2020)"""):
        st.write(text_tpnilm)
        st.image("Figures/TPNILM.png", caption="TPNILM architecture.")

    with st.expander(f"""### Unet-NILM (2020)"""):
        st.write(text_unetnilm)
        st.image("Figures/Unet-NILM.png", caption="Unet-NILM architecture.")

    with st.expander(f"""### TransNILM (2023)"""):
        st.write(text_transnilm)
        st.image("Figures/TransNILM.png", caption="TransNILM architecture.")

    with st.expander(f"""### BiGRU (2023)"""):
        st.write(text_bigru)
        st.image("Figures/BIGRU.png", caption="BiGRU architecture.")

    with st.expander(f"""### CRNN (2024)"""):
        st.write(text_crnn)
        st.image("Figures/CRNN.png", caption="CRNN architecture.")






with tab_dataset_description:
    st.markdown("""### Smart meters datasets""")
    st.markdown(text_description_dataset)

