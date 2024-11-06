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

col1_1, col1_2, col1_3 = st.columns(3)

with col1_1:
        dataset_name = st.selectbox(
            "Choose a dataset", list_dataset, index=0
        )

with col1_2:
    measure_detection = st.selectbox(
        "Choose a detection measure:", measures_list, index=1
    )

with col1_3:
    measure_localization = st.selectbox(
        "Choose a localization measure:", measures_list, index=2
    )

dict_measure = {'Accuracy': 'ACCURACY', 'Balanced Accuracy': 'BALANCED_ACCURACY', 
                'F1 Score': 'F1_SCORE', 'Precision': 'PRECISION', 'Recall': 'RECALL'}
measure_detection    = dict_measure[measure_detection]
measure_localization = dict_measure[measure_localization]

tab_playground, tab_benchmark = st.tabs(
        ["CamAL performances", "Comparaison with NILM-based (SeqToSeq) approaches"]
    )
    
with tab_playground:
    st.markdown("### CamAL performance")

    df_res_bench = get_bench_results(dataset_name)

    st.markdown("Detection and localization score for the appliances")

    fig_bench_detection    = plot_detection_score_for_dataset(df_res_bench, measure_detection)
    st.plotly_chart(fig_bench_detection, use_container_width=True)

    fig_bench_localization = plot_localization_score_for_dataset(df_res_bench, measure_localization)
    st.plotly_chart(fig_bench_localization, use_container_width=True)

    st.markdown("Influence of the window length used for training by CamAL")

    fig_influence_win_train =  plot_influence_win_train(df_res_bench, measure_detection, measure_localization)

    st.plotly_chart(fig_influence_win_train, use_container_width=True)




with tab_benchmark:

    dict_ts_device = {'UKDALE': devices_list_refit_ukdale,
                      'REFIT': devices_list_refit_ukdale,
                      'IDEAL': devices_list_ideal}
    
    appliance_selected = st.selectbox(
        "Select an appliance:", dict_ts_device[dataset_name], index=0
    )

    st.markdown("""### CamAL performance VS SotA NILM methods""")
    st.markdown("Comparaison of CamAL to weakly and fully supervised NILM methods""")
    df_res = get_bench_results_nilm(dataset_name)
    fig_perf_comparaison = plot_nilm_performance_comparaison(df_res, dataset_name, appliance_selected, measure_localization)
    st.plotly_chart(fig_perf_comparaison, use_container_width=True)


    st.markdown("""### Vizualise the results with NILM based model""")

    col3_1, col3_2 = st.columns(2)

    with col3_1:
        length = st.selectbox(
            "Choose the window length:", lengths_list, index=2
        )
    with col3_2:
        dict_ts_list = {'UKDALE': list_ukdale_ts,
                        'REFIT': list_refit_ts,
                        'IDEAL': list_ideal_ts}
        ts_name = st.selectbox(
            "Choose a load curve", dict_ts_list[dataset_name], index=0
        )


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
        
    
    pred_prob_flag = st.toggle('Display probabilities instead of status')
    
    pred_nilmcam         = pred_one_window_nilmcam(st.session_state.CURRENT_WINDOW_BENCHMARK, df, window_size, dataset_name, [appliance_selected])[appliance_selected]
    fig_visu_comparaison = plot_one_window_benchmark(st.session_state.CURRENT_WINDOW_BENCHMARK, df, window_size, appliance_selected, pred_nilmcam, pred_prob_flag)
    st.plotly_chart(fig_visu_comparaison, use_container_width=True)
