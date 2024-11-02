"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st

from utils import *
from constants import *
    
st.markdown(text_tab_playground)

col1_1, col1_2, col1_3 = st.columns(3)

with col1_1:
    ts_name = st.selectbox(
        "Choose a load curve", list_name_ts, index=0
    )
with col1_2:
    length = st.selectbox(
        "Choose the window length:", lengths_list, index=2
    )
with col1_3:
    appliances1 = st.multiselect(
        "Choose devices:", devices_list_ideal if 'IDEAL' in ts_name else devices_list_refit_ukdale,
    )


models    = ['ResNetEnsemble']

st.write(CURRENT_WINDOW)

colcontrol_1, colcontrol_2, colcontrol_3 = st.columns([0.2,0.8,0.2])
with colcontrol_1:
    if st.button(":rewind: **Prev.**", type="primary"):
        CURRENT_WINDOW -= 1
with colcontrol_3:
    if st.button("**Next** :fast_forward:", type="primary"):
        CURRENT_WINDOW += 1

df, window_size = get_time_series_data(ts_name, length=length)
n_win = len(df) // window_size
st.write(n_win)

if CURRENT_WINDOW > n_win:
    CURRENT_WINDOW=0
elif CURRENT_WINDOW < 0:
    CURRENT_WINDOW=n_win

st.write(CURRENT_WINDOW)

with colcontrol_2:
    st.markdown("<p style='text-align: center;'> <b>from</b> <i>{}</i> <b>to</b> <i>{}</i> </p>".format(df.iloc[CURRENT_WINDOW*window_size: (CURRENT_WINDOW+1)*window_size].index[0],df.iloc[CURRENT_WINDOW*window_size: (CURRENT_WINDOW+1)*window_size].index[-1]),unsafe_allow_html=True)
    
if len(appliances1)>0:
    if len(models)>0:
        pred_dict_all              = pred_one_window(CURRENT_WINDOW, df, window_size, ts_name, appliances1, models)
        fig_ts, fig_app, fig_stack = plot_one_window3(CURRENT_WINDOW,  df, window_size, appliances1, pred_dict_all)
            
        fig_prob = plot_detection_probabilities(pred_dict_all)
        
        tab_ts, tab_app = st.tabs(["Aggregated", "Per device"])
        
        with tab_ts:
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with tab_app:
            on = st.toggle('Stack')
            if on:
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.plotly_chart(fig_app, use_container_width=True)


        tab_prob, tab_signatures = st.tabs(["Models detection probabilities", "Examples of appliance patterns"])

        with tab_prob:
            st.plotly_chart(fig_prob, use_container_width=True)

        with tab_signatures:
            fig_sig = plot_signatures(appliances1)
            st.plotly_chart(fig_sig, use_container_width=True)

    else:
        fig_ts, fig_app, fig_stack = plot_one_window2(CURRENT_WINDOW,  df, window_size, appliances1)

        tab_ts, tab_app = st.tabs(["Aggregated", "Per device"])

        with tab_ts:
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with tab_app:
            on = st.toggle('Stack')
            if on:
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.plotly_chart(fig_app, use_container_width=True)
        
        fig_sig = plot_signatures(appliances1)

        st.plotly_chart(fig_sig, use_container_width=True)
else:
    fig_ts = plot_one_window_agg(CURRENT_WINDOW,  df, window_size)
    
    st.plotly_chart(fig_ts, use_container_width=True)

    all_appliances = ['WashingMachine', 'Dishwasher', 'Microwave', 'Kettle', 'Shower']
    fig_sig = plot_signatures(all_appliances)
    st.plotly_chart(fig_sig, use_container_width=True)
