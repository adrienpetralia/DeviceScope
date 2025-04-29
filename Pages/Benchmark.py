"""
@who:
@where: 
@when: 
@what: 
"""

import streamlit as st


from Utils.utils import *
from Utils.constants import *

config = {
  'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'image',
    'scale': 1
  }
}

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

tab_comparaison_with_nilm, tab_camal_performances = st.tabs(
        ["Comparison with SotA NILM approach", "CamAL performance"]
    )
    
with tab_comparaison_with_nilm:
    dict_ts_device = {'UKDALE': devices_list_refit_ukdale,
                      'REFIT': devices_list_refit_ukdale,
                      'IDEAL': devices_list_ideal}
    
    appliance_selected = st.selectbox(
        "Select an appliance:", dict_ts_device[dataset_name], index=0
    )

    st.markdown("""### CamAL performance VS SotA NILM methods""")
    st.markdown("""On this tab, you can interactively compare the performance of CamAL against SotA methods proposed for appliance detection. First, take a look at the **performances** (for the chosen metric) according to the number of labels used for training the methods. Then, **compare visually** the prediction of each baseline for the select dataset and appliance. Don't forget that **CamAL** (as well as the weak version of the CRNN) used only **one label** per window for training (one label per house for our method on the IDEAL dataset), whereas all the other baselines used **one label per timestamp**!""")
    st.markdown("##### 1. Accuracy vs Number of label used for training""")
    df_res = get_bench_results_nilm(dataset_name)
    fig_perf_comparaison = plot_nilm_performance_comparaison(df_res, dataset_name, appliance_selected, measure_localization)
    #fig_perf_comparaison = plot_nilm_performance_comparaison_trainingtime_vs_accuracy(df_res, dataset_name, appliance_selected, measure_localization)
    st.plotly_chart(fig_perf_comparaison, use_container_width=True, config=config)


    st.markdown("""##### 2. Compare visually the prediction of the different approaches""")

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
    st.plotly_chart(fig_visu_comparaison, use_container_width=True, config=config)


with tab_camal_performances:
    st.markdown("### CamAL performance")
    st.markdown("""On this tab, you can interactively explore CamAL's performance for the different datasets. First, take a look at the detection and localization scores for the different appliances. Then, notice the influence of the window length (that plays a role in the number of labels available for training) on the performance!""")
    

    df_res_bench = get_bench_results(dataset_name)

    st.markdown("##### 1. Detection and localization score")

    fig_bench_detection    = plot_detection_score_for_dataset(df_res_bench, measure_detection)
    st.plotly_chart(fig_bench_detection, use_container_width=True, config=config)

    fig_bench_localization = plot_localization_score_for_dataset(df_res_bench, measure_localization)
    st.plotly_chart(fig_bench_localization, use_container_width=True, config=config)

    st.markdown("##### 2. Influence of the window length on the performance")

    fig_influence_win_train =  plot_influence_win_train(df_res_bench, measure_detection, measure_localization)

    st.plotly_chart(fig_influence_win_train, use_container_width=True, config=config)
