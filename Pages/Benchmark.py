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

dataset_name = 'IDEAL'

with col1_2:
    measure_detection = st.selectbox(
        "Choose a detection measure:", measures_list, index=0
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
        ["NILM-CAM performances", "Comparaison with NILM approaches"]
    )
    
with tab_playground:
    st.markdown("### Appliance detection performance comparaison")

    df_res_bench = get_bench_results(dataset_name)
    num_apps = df_res_bench['Case'].nunique()

    # Create a subplot figure with two columns and one row for each appliance
    fig = make_subplots(
        rows=1, cols=2, 
        horizontal_spacing=0.1
    )

    # Loop through each appliance and add its corresponding subplots
    for i, app in enumerate(df_res_bench['Case'].unique(), start=1):
        df = df_res_bench.loc[df_res_bench['Case'] == app]
    
        # Detection Metric Plot (Clf_F1_SCORE)
        win_df_clf = df.groupby(['Win', 'WinTrainWeak'])[f'Clf_{measure_localization}'].mean().reset_index()

        fig.add_trace(
            go.Scatter(
                x=win_df_clf['WinTrainWeak'], 
                y=win_df_clf[f'Clf_{measure_detection}'], 
                mode='lines', 
                name=f'{app}',
                legendgroup=f'{app}',  # Group by the same name to share color
                marker_color=dict_color_appliance[app],
                showlegend=True
            ),
            row=1, col=1
        )

        # Classification Metric Plot (F1_SCORE)
        win_df_clf = df.groupby(['Win', 'WinTrainWeak'])[measure_localization].mean().reset_index()

        fig.add_trace(
            go.Scatter(
                x=win_df_clf['WinTrainWeak'], 
                y=win_df_clf[measure_localization], 
                mode='lines', 
                name=f'{app}',
                legendgroup=f'{app}',  # Group by the same name to share color
                marker_color=dict_color_appliance[app],
                showlegend=False
            ),
            row=1, col=2
        )

    # Update the layout of the figure
    fig.update_layout(
        height=400,  # Adjust the height based on the number of rows (each appliance gets two columns)
        title=f'Accuracy vs WinTrainWeak for Different Appliances (Detection and Classification)',
        showlegend=True
    )

    # Update axes titles for all subplots
    fig.update_xaxes(title_text="WinTrainWeak", row=1, col=1)
    fig.update_yaxes(title_text=f'Clf_{measure_detection}', row=1, col=1)
    fig.update_xaxes(title_text="WinTrainWeak", row=1, col=2)
    fig.update_yaxes(title_text=f'{measure_localization}', row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)




with tab_benchmark:

    dict_ts_device = {'UKDALE': devices_list_refit_ukdale,
                      'REFIT': devices_list_refit_ukdale,
                      'IDEAL': devices_list_ideal}
    
    appliance_selected = st.selectbox(
        "Select an appliance:", dict_ts_device[dataset_name], index=0
    )


    st.markdown("""### Applicance pattern localization performances compared to other approach according to the number of label used for training""")
    df_res = get_bench_results_nilm(dataset_name)
    fig_perf_comparaison = plot_nilm_performance_comparaison(df_res, dataset_name, appliance_selected, measure_localization)
    st.plotly_chart(fig_perf_comparaison, use_container_width=True)


    st.markdown("""### Compare the results with NILM based model""")

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
