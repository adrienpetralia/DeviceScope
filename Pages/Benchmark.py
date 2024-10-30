"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st
from constants import *

from utils import *

st.markdown(text_tab_benchmark)

col1, col2 = st.columns(2)

with col1:
    measure = st.selectbox(
        "Choose metric", measures_list, index=0
    )
with col2:
    dataset = st.selectbox(
        "Choose dataset", dataset_list, index=0
    )

#st.markdown("#### Overall results")

fig1 = plot_benchmark_figures1(measure, dataset)
fig2 = plot_benchmark_figures2(measure, dataset)
fig3 = plot_benchmark_figures3(measure, dataset)

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

#st.markdown("#### Explore the influence of the sampling rate on the detection performance for selected appliance(s).")

# appliances2 = st.multiselect(
#     "Select devices:", devices_list_refit_ukdale,
# )

# fig_benchmark = plot_benchmark_figures4(appliances2, measure, dataset)
# st.plotly_chart(fig_benchmark, use_container_width=True)
