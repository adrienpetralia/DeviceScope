"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st


from Utils.utils import *
from Utils.constants import *

# Use session state to store CURRENT_WINDOW_BENCHMARK to persist across user interactions
if 'CURRENT_WINDOW_BENCHMARK' not in st.session_state:
    st.session_state.CURRENT_WINDOW_BENCHMARK = 0

st.markdown(text_tab_benchmark)

tab_playground, tab_benchmark = st.tabs(
        ["Performances", "Comparaison with NILM approaches"]
    )
    
with tab_playground:
    run_metric_comparaison_frame()

with tab_benchmark:
    col1_1, col1_2 = st.columns(2)

    with col1_1:
        dataset_name = st.selectbox(
            "Choose a load curve", list_dataset, index=0
        )

    with col1_2:
        if dataset_name=='UKDALE':
            list_name_ts = list_ukdale_ts
        elif dataset_name=='REFIT':
            list_name_ts = list_refit_ts
        elif dataset_name=='IDEAL':
            list_name_ts = list_ukdale_ts
        else:
            raise ValueError('Wrong dataset name.')
        ts_name = st.selectbox(
            "Choose a load curve", list_name_ts, index=0
        )

    col2_1, col2_2 = st.columns(2)

    with col2_1:
        length = st.selectbox(
            "Choose the window length:", lengths_list, index=2
        )
    with col2_2:
        appliance_selected = st.selectbox(
            "Choose devices:", devices_list_ideal if dataset_list=='IDEAL' else devices_list_refit_ukdale, index=0
        )
    #appliance_selected = 'Dishwasher'

    st.markdown("""### Applicance localization performance comparaison according the number of label used for training""")
    fig_perf_comparaison = plot_nilm_performance_comparaison('IDEAL', 'Dishwasher', 'F1_SCORE')
    st.plotly_chart(fig_perf_comparaison, use_container_width=True)


    st.markdown("""### Compare the results with NILM based model""")
    colcontrol_1, colcontrol_2, colcontrol_3 = st.columns([0.2, 0.8, 0.2])
    with colcontrol_1:
        if st.button(":rewind: **Prev.**", type="primary"):
            st.session_state.CURRENT_WINDOW_BENCHMARK -= 1
    with colcontrol_3:
        if st.button("**Next** :fast_forward:", type="primary"):
            st.session_state.CURRENT_WINDOW_BENCHMARK += 1

    # Load the time series data
    df, window_size = get_pred_data(os.getcwd()+f'/Pred/{dataset_name}/{appliance_selected}/{ts_name}.gzip', length=length)
    n_win = len(df) // window_size

    # Ensure CURRENT_WINDOW_BENCHMARK stays within valid bounds
    if st.session_state.CURRENT_WINDOW_BENCHMARK >= n_win:
        st.session_state.CURRENT_WINDOW_BENCHMARK = 0
    elif st.session_state.CURRENT_WINDOW_BENCHMARK < 0:
        st.session_state.CURRENT_WINDOW_BENCHMARK = n_win - 1

    # Display window range
    with colcontrol_2:
        st.markdown("<p style='text-align: center;'> <b>from</b> <i>{}</i> <b>to</b> <i>{}</i> </p>".format(
            df.iloc[st.session_state.CURRENT_WINDOW_BENCHMARK * window_size: (st.session_state.CURRENT_WINDOW_BENCHMARK + 1) * window_size].index[0],
            df.iloc[st.session_state.CURRENT_WINDOW_BENCHMARK * window_size: (st.session_state.CURRENT_WINDOW_BENCHMARK + 1) * window_size].index[-1]),
            unsafe_allow_html=True)
        
    
    
    #pred_nilmcam    = pred_one_window_nilmcam(st.session_state.CURRENT_WINDOW_BENCHMARK, pred, window_size, dataset_name, [appliance_selected])
    pred_nilmcam = 0
    
    fig_visu_comparaison = plot_one_window_benchmark(st.session_state.CURRENT_WINDOW_BENCHMARK, df, window_size, appliance_selected, pred_nilmcam)
    pred_status_flag = st.toggle('Show probabilities')
    st.plotly_chart(fig_visu_comparaison, use_container_width=True)
