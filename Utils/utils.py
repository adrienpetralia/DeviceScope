import streamlit as st

# === Lib import === #
import os, lzma, io, pickle
import numpy as np
import pandas as pd
import torch

# === Vizu import === #
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# === Customs import === #
from Utils.constants import *
#from Utils.constants import *

from Models.Classifiers.ResNet3 import ResNet3
from Models.Classifiers.ResNet3LN import ResNet3LN
from Models.Classifiers.ResNet5 import ResNet5
from Models.Classifiers.ResNet5LN import ResNet5LN

from Helpers.class_activation_map import CAM
        

def plot_detection_score_for_dataset(df_res_bench, measure_detection):
    # Calculate the average Clf_F1_SCORE for each appliance (Case) across different seeds
    #average_f1_score = df_res_bench.loc[df_res_bench['WinTrainWeak']==10080]
    average_f1_score = df_res_bench.groupby(['Case']).max(numeric_only=True).reset_index()

    # Plotting the average Clf_F1_Score for each appliance
    fig = px.bar(average_f1_score, 
                 x='Case', 
                 y=f'Clf_{measure_detection}',
                 color='Case',
                 color_discrete_map=dict_color_appliance,
                 title=f'Detection {dict_measure_to_display[measure_detection]} for the Appliances available in the selected dataset',
                 labels={f'Clf_{measure_detection}': f'Average {dict_measure_to_display[measure_detection]}', 'Case': 'Appliance'},
                 text=f'Clf_{measure_detection}')

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(title=f'Average {dict_measure_to_display[measure_detection]}'), xaxis=dict(title='Appliance'))

    return fig

def plot_localization_score_for_dataset(df_res_bench, measure_localization):
    # Calculate the average Clf_F1_SCORE for each appliance (Case) across different seeds
    #average_f1_score = df_res_bench.loc[df_res_bench['WinTrainWeak']==10080]
    average_f1_score = df_res_bench.groupby(['Case']).max(numeric_only=True).reset_index()

    # Plotting the average Clf_F1_Score for each appliance
    fig = px.bar(average_f1_score, 
                 x='Case', 
                 y=f'{measure_localization}',
                 color='Case',
                 color_discrete_map=dict_color_appliance,
                 title=f'Localization {dict_measure_to_display[measure_localization]} for the Appliances available in the selected dataset',
                 labels={f'{measure_localization}': f'Average {dict_measure_to_display[measure_localization]}', 'Case': 'Appliance'},
                 text=measure_localization)

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(title=f'Average {dict_measure_to_display[measure_localization]}'), xaxis=dict(title='Appliance'))

    return fig

def plot_influence_win_train(df_res_bench, measure_detection, measure_localization):
    # Create a subplot figure with two columns and one row for each appliance
    fig = make_subplots(
        rows=1, cols=2, 
        horizontal_spacing=0.2
    )

    # Loop through each appliance and add its corresponding subplots
    for i, app in enumerate(df_res_bench['Case'].unique(), start=1):
        df_res_bench_app = df_res_bench.loc[df_res_bench['Case']==app]
    
        # Detection Metric Plot (Clf_F1_SCORE)
        win_df_clf = df_res_bench_app.groupby(['Win', 'WinTrainWeak'])[f'Clf_{measure_detection}'].mean().reset_index()

        fig.add_trace(
            go.Scatter(
                x=win_df_clf['WinTrainWeak'], 
                y=win_df_clf[f'Clf_{measure_detection}'], 
                mode='lines+markers',
                name=f'{app}',
                legendgroup=f'{app}',  # Group by the same name to share color
                marker_color=dict_color_appliance[app],
                showlegend=True
            ),
            row=1, col=1
        )

        # Classification Metric Plot (F1_SCORE)
        win_df_clf = df_res_bench_app.groupby(['Win', 'WinTrainWeak'])[measure_localization].mean().reset_index()

        fig.add_trace(
            go.Scatter(
                x=win_df_clf['WinTrainWeak'], 
                y=win_df_clf[measure_localization], 
                mode='lines+markers',
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
        title=f'Accuracy vs legnth of the window used for training',
        showlegend=True
    )

    # Update axes titles for all subplots
    fig.update_xaxes(title_text="Window length used for training", row=1, col=1)
    fig.update_yaxes(title_text=f'Detection {dict_measure_to_display[measure_detection]}', row=1, col=1)
    fig.update_xaxes(title_text="Window length used for training", row=1, col=2)
    fig.update_yaxes(title_text=f'Localization {dict_measure_to_display[measure_detection]}', row=1, col=2)

    return fig



def get_dataset_name(ts_name):
    # Get dataset_name according to choosen ts_name
    if 'UKDALE' in ts_name:
        return 'UKDALE'
    elif 'REFIT' in ts_name:
        return 'REFIT'
    elif 'IDEAL' in ts_name:
        return 'IDEAL'
    else:
        raise ValueError('Wrong dataset name.')


def convert_length_to_window_size(length):
    length_to_minutes = {
        '6 hours': 360,
        '12 hours': 720,
        '1 Day': 1440
    }
    
    if length in length_to_minutes:
        return length_to_minutes[length]
    else:
        raise ValueError("Length not recognized. Please use '6 hours', '12 hours', or '1 Day'.")
    

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def get_pred_data(file, length):
    # Convert selected length to window_size
    window_size = convert_length_to_window_size(length)
    data = pd.read_csv(file, parse_dates=['Time'], index_col=['Time'], compression='gzip')
    return data, window_size

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def get_time_series_data(ts_name, length):
    # Convert selected length to window_size
    window_size = convert_length_to_window_size(length)

    # Load dataframe
    df = pd.read_csv(os.getcwd()+f'/Data/{ts_name}.gzip', compression='gzip', parse_dates=['Time']).set_index('Time')

    return df, window_size

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def get_bench_results_nilm(dataset):
    # Load dataframe
    df = pd.read_csv(os.getcwd()+f'/TableResults/{dataset}BenchNILMResults.gzip', compression='gzip')
    df.replace('CRNNWeak', 'CRNN (Weak)', inplace=True)
    df.replace('CRNNStrong', 'CRNN', inplace=True)
    df.replace('Unet-NILM', 'UNet-NILM', inplace=True)

    return df

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def get_bench_results(dataset):
    # Load dataframe
    df = pd.read_csv(os.getcwd()+f'/TableResults/{dataset}BenchResults.gzip', compression='gzip')

    return df


def get_resnet_instance(resnet_name, kernel_size, **kwargs):
    if resnet_name =='ResNet3':
        inst = ResNet3(kernel_sizes=[kernel_size, 7, 3], **kwargs)
    elif resnet_name =='ResNet3LN':
        inst = ResNet3LN(kernel_sizes=[kernel_size, 7, 3], **kwargs)
    elif resnet_name=='ResNet5':
        inst = ResNet5(kernel_sizes=[kernel_size, 7, 3], **kwargs)
    elif resnet_name =='ResNet5LN':
        inst = ResNet5LN(kernel_sizes=[kernel_size, 7, 3], **kwargs)
    else:
        raise ValueError('ResNet name {} unknown'.format(resnet_name))

    return inst

def get_resnet_layers(resnet_name, resnet_inst):
    if 'ResNet3' in resnet_name:
        last_conv_layer = resnet_inst._modules['layers'][2]
        fc_layer_name   = resnet_inst._modules['linear']
    elif 'ResNet5' in resnet_name:
        last_conv_layer = resnet_inst._modules['layers'][4]
        fc_layer_name   = resnet_inst._modules['linear']
    else:
        raise ValueError('ResNet name {} unknown'.format(resnet_name))

    return last_conv_layer, fc_layer_name

def get_soft_label_ensemble(current_win, path_ensemble_clf):
    resnet_type = 'ResNet3'
    device      = 'cpu'

    with open(f'{path_ensemble_clf}LogResNetsEnsemble.pkl', 'rb') as handle:
        dict_results = pickle.load(handle)

    list_best_resnets = dict_results['ListBestResNets']

    current_win = current_win / 1000
    soft_label = np.zeros_like(current_win)
    
    prob_detect = 0

    # Loop on BestResNets 
    for resnet_name in list_best_resnets:
        resnet_inst = get_resnet_instance(resnet_type, dict_results[resnet_name]['kernel_size'])
        #resnet_inst.to(device)

        if os.path.exists(f'{path_ensemble_clf}{resnet_name}.xz'):
            path_model = f'{path_ensemble_clf}{resnet_name}.xz'
            with lzma.open(path_model, 'rb') as file:
                decompressed_file = file.read()
            log = torch.load(io.BytesIO(decompressed_file), map_location=torch.device(device))
            del decompressed_file
        else:
            raise ValueError(f'Provide folders {path_ensemble_clf} does not contain {resnet_name} clf.')

        resnet_inst.load_state_dict(log['model_state_dict'])
        resnet_inst.eval()
        last_conv_layer, fc_layer_name = get_resnet_layers(resnet_type, resnet_inst)

        CAM_builder = CAM(model=resnet_inst, device=device, 
                          last_conv_layer=last_conv_layer, fc_layer_name=fc_layer_name)
        cam, y_pred, proba = CAM_builder.run(instance=current_win, returned_cam_for_label=1)
        prob_detect += proba[1]

        if y_pred>0:
            # Clip CAM and MaxNormalization (between 0 and 1)
            clip_cam = np.nan_to_num(np.clip(cam, a_min=0, a_max=None).astype(np.float16), nan=0.0, neginf=0.0, posinf=0.0)

            if clip_cam.max()>0:
                clip_cam = clip_cam / clip_cam.max()
            else:
                clip_cam = np.zeros_like(clip_cam)

            soft_label = soft_label + clip_cam.ravel()

        del resnet_inst
    
    # Majority voting: if appliance not detected in current win, soft label set to 0
    # if (y is not None) or ((prob_detect / len(list_best_resnets)) >= 0.5):
    prob_detect = prob_detect / len(list_best_resnets)

    if prob_detect >= 0.5:
        soft_label = soft_label / len(list_best_resnets)

        # Small moving average 
        soft_label = moving_average(soft_label, w=5)
        avg_cam    = np.copy(soft_label)

        # Sigmoid-Attention between input aggregate power and computed avg. CAM score
        soft_label = soft_label * current_win
        soft_label_before_sig = np.copy(soft_label)
        soft_label = sigmoid(soft_label)
        soft_label = np.round(soft_label)

        return prob_detect, soft_label, soft_label_before_sig, avg_cam
    else: 
        return prob_detect, np.zeros_like(current_win), np.zeros_like(current_win), np.zeros_like(current_win)

    


def sigmoid(z):
    return 2 * (1.0/(1.0 + np.exp(-z))) - 1


def sigmoid_charge(z, mean_cdc=0.3, std_cdc=0.45):
    return  (1.0/(1.0 + np.exp(-z + mean_cdc + 2 * std_cdc)))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def get_pred_nilmcam_one_appliance(dataset_name, window_agg, appliance):

    path_ensemble = os.getcwd()+f'/TrainedModels/{dataset_name}/1min/{appliance}/ResNetEnsemble/'
    pred_prob, soft_label, soft_label_before_sig, avg_cam = get_soft_label_ensemble(window_agg, path_ensemble)
    
    # if appliance=='Microwave':
    #     soft_label = sigmoid(np.array(soft_label_before_sig))
    #     soft_label = np.where(soft_label > 0.2, 1, 0)

    # elif appliance=='WashingMachine':
    #     soft_label = sigmoid(np.array(soft_label_before_sig))
    #     soft_label = np.where(soft_label > 0.2, 1, 0)

    return {'pred_prob': pred_prob, 'pred_status': soft_label, 'soft_label_before_sig': soft_label_before_sig, 'avg_cam': avg_cam}



def pred_one_window_nilmcam(k, df, window_size, dataset_name, appliances):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    window_agg = window_df['Aggregate']

    pred_dict = {}
    for appl in appliances:   
        pred_dict[appl]  = get_pred_nilmcam_one_appliance(dataset_name, window_agg, appl)

    return pred_dict


def plot_one_window_agg(k, df, window_size):
    window_df = df.iloc[k*window_size: k*window_size + window_size]

    fig_agg          = go.Figure()
    
    # Aggregate plot
    fig_agg.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', fill='tozeroy', line=dict(color='royalblue')))
    
    # Update layout for the combined figure
    fig_agg.update_layout(
        title='Aggregate power consumption',
        xaxis_title='Time',
        height=300,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )
    
    # Update y-axis for the aggregate consumption plot
    fig_agg.update_yaxes(title_text='Power (Watts)', range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])

    return fig_agg


def plot_one_window_playground(k, df, window_size, appliances, pred_dict_all_appliance):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    
    # Create subplots with 2 rows, shared x-axis
    list_row_heights = [0.6] + [0.4/len(appliances) for _ in range(len(appliances))]

    fig_agg          = make_subplots(rows=len(appliances)+1, cols=1, 
                                     shared_xaxes=True, vertical_spacing=0.1, row_heights=list_row_heights,
                                     subplot_titles=[""]+appliances)
    fig_appl         = make_subplots(rows=len(appliances)+1, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.1, row_heights=list_row_heights,
                                     subplot_titles=[""]+appliances)
    
    # Aggregate plot
    fig_agg.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', fill='tozeroy', line=dict(color='royalblue')),
                      row=1, col=1)
    
    # Stacked CAM calculations
    for z, appl in enumerate(appliances, start=1):

        fig_appl.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', name=appl.capitalize(), marker_color=dict_color_appliance[appl],  fill='tozeroy'))

        pred_dict_app = pred_dict_all_appliance[appl]

        #if label_to_display=='status':
        # pred_nilmcam_app = pred_dict_app['soft_label_before_sig']
        pred_nilmcam_app = pred_dict_app['pred_status']
        
        fig_agg.add_trace(go.Scatter(x=window_df.index, y=pred_nilmcam_app, mode='lines', showlegend=False, name=appl.capitalize(), marker_color=dict_color_appliance[appl], fill='tozeroy'), row=1+z, col=1)
        fig_appl.add_trace(go.Scatter(x=window_df.index, y=pred_nilmcam_app, mode='lines', showlegend=False,  name=appl.capitalize(), marker_color=dict_color_appliance[appl],  fill='tozeroy'), row=1+z, col=1)


        color = dict_color_appliance[appl]  # Get color for the current appliance        

        if appl=='WashingMachine' or appl=='Dishwasher':
            w=30
        else:
            w=15

        pred_nilmcam_app = np.convolve(pred_nilmcam_app, np.ones(w), 'same') / w

        threshold = 0
        start_idx = None 

        for i, value in enumerate(pred_nilmcam_app):
            if value > threshold and start_idx is None:  # CAM becomes active
                start_idx = i
            elif value <= threshold and start_idx is not None:  # End of an active segment
                # Add shape for the active segment
                fig_agg.add_shape(
                    type="rect",
                    x0=window_df.index[start_idx],  # Convert index to x-value as needed
                    y0=0,
                    x1=window_df.index[i],
                    y1=max(3000, np.max(window_df['Aggregate'].values) + 50),
                    line=dict(width=0),
                    fillcolor=color,
                    opacity=0.3,  # Adjust for desired transparency
                    layer="below",
                    row=1, col=1
                )
                start_idx = None  # Reset for next segment

        # Check if there's an active segment until the end
        if start_idx is not None:
            fig_agg.add_shape(
                type="rect",
                x0=window_df.index[start_idx],
                y0=0,
                x1=window_df.index[-1],
                y1=max(3000, np.max(window_df['Aggregate'].values) + 50),
                line=dict(width=0),
                fillcolor=color,
                opacity=0.3,
                layer="below",
                row=1, col=1
            )

    # Update layout for the combined figure
    xaxis_title_dict = {f'xaxis{len(appliances)+1}_title': 'Time'}
    fig_agg.update_layout(
        title='Aggregate power consumption and predicted appliance localization',
        showlegend=False,
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40),
        **xaxis_title_dict
    )
    
    fig_appl.update_layout(
        title='Individual appliance power consumption compared to predicted appliance localization',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40),
        **xaxis_title_dict
    )
    
    fig_agg.update_annotations(font=dict(family="Helvetica", size=15))
    fig_appl.update_annotations(font=dict(family="Helvetica", size=15))

    fig_agg.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
   
    # Update y-axis for the heatmap
    for z, appl in enumerate(appliances, start=2):
        fig_agg.update_yaxes(row=z, col=1, range=[0, 1], visible=False, showticklabels=False)
        fig_appl.update_yaxes(row=z, col=1, range=[0, 1], visible=False, showticklabels=False)

    if len(appliances)==4:
        yaxis_title_y = 0.3
    elif len(appliances)==3:
        yaxis_title_y = 0.27
    elif len(appliances)==3:
        yaxis_title_y = 0.25
    else:
        yaxis_title_y = 0.22
        
    shared_yaxis_title = {
        'text': "Pred. Status",  # Update with your desired title
        'showarrow': False,
        'xref': 'paper',
        'yref': 'paper',
        'x': -0.05,
        'y': yaxis_title_y,
        'xanchor': 'center',
        'yanchor': 'middle',
        'textangle': -90,  # Rotate the text for vertical alignment
        'font': {'size': 15}
    }

    for fig in [fig_agg, fig_appl]:
        if 'annotations' in fig.layout:
            fig.layout.annotations += (shared_yaxis_title,)
        else:
            fig.update_layout(annotations=[shared_yaxis_title])

    return fig_agg, fig_appl



def plot_one_window_benchmark(k, df, window_size, appliance, pred_nilm_cam, pred_prob_flag):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    
    to_plot = list(df.columns)


    # Create subplots, shared x-axis
    list_row_heights = [0.3] + [0.7/8 for _ in range(8)]

    fig = make_subplots(rows=9, cols=1, 
                        shared_xaxes=True, #vertical_spacing=0.1
                        row_heights=list_row_heights,
                        subplot_titles=['', f'{appliance} Status', 'CamAL', 'CRNN (Weak)', 'BiGRU', 'UNet-NILM', 'TPNILM', 'TransNILM', 'CRNN'])
    
    # Aggregate plot
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], 
                             mode='lines', 
                             showlegend=False,
                             name='Aggregate', fill='tozeroy', 
                             line=dict(color='royalblue')),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df[appliance], 
                             mode='lines', 
                             showlegend=False,
                             name=appliance, fill='tozeroy', 
                             line=dict(color=dict_color_appliance[appliance])),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=window_df.index, y=window_df[appliance+'_Status'], 
                             mode='lines', 
                             showlegend=False,
                             name=appliance, fill='tozeroy', 
                             line=dict(color=dict_color_appliance[appliance])),
                  row=2, col=1)
    

    fig.add_trace(go.Scatter(x=window_df.index, y=pred_nilm_cam['pred_status'] if not pred_prob_flag else pred_nilm_cam['avg_cam'], 
                             mode='lines', 
                             showlegend=False, name='CamAL',
                             line=dict(color=dict_color_model['CamAL']),
                             fill='tozeroy'), 
                    row=3, col=1)


    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['CRNNWeak'].round() if not pred_prob_flag else window_df['CRNNWeak'], 
                             mode='lines',
                             line=dict(color=dict_color_model['CRNN (Weak)']),
                             showlegend=False, name='CRNN (Weak)', 
                             fill='tozeroy'), 
                    row=4, col=1)


    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['BiGRU'].round() if not pred_prob_flag else window_df['BiGRU'], 
                             mode='lines', 
                             line=dict(color=dict_color_model['BiGRU']),
                             showlegend=False, name='BiGRU', 
                             fill='tozeroy'), 
                    row=5, col=1)
    
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['UNET_NILM'].round() if not pred_prob_flag else window_df['UNET_NILM'], 
                               mode='lines', 
                               showlegend=False, name='UNet-NILM', 
                               line=dict(color=dict_color_model['UNet-NILM']),
                               fill='tozeroy'), 
                    row=6, col=1)
    
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['TPNILM'].round() if not pred_prob_flag else window_df['TPNILM'], 
                             mode='lines', 
                             line=dict(color=dict_color_model['TPNILM']),
                                showlegend=False, name='TPNILM', 
                                fill='tozeroy'), 
                    row=7, col=1)
    
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['TransNILM'].round() if not pred_prob_flag else window_df['TransNILM'], 
                             mode='lines', 
                                showlegend=False, name='TransNILM', 
                                line=dict(color=dict_color_model['TransNILM']),
                                fill='tozeroy'), 
                    row=8, col=1)
    
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df['CRNNStrong'].round() if not pred_prob_flag else window_df['CRNNStrong'], 
                             mode='lines', 
                             showlegend=False, name='CRNN (Strong)', 
                             line=dict(color=dict_color_model['CRNN']),
                             fill='tozeroy'), 
                    row=9, col=1)


    # Update layout for the combined figure
    xaxis_title_dict = {f'xaxis{9}_title': 'Time'}
    fig.update_layout(
        title='Aggregate power consumption and predicted appliance localization',
        showlegend=False,
        height=1000,
        margin=dict(l=50, r=20, t=20, b=20),
        **xaxis_title_dict
    )
    
    
    fig.update_annotations(font=dict(family="Helvetica", size=15))

    fig.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
 
    # Update y-axis for the heatmap
    for z in range(2, 10):
        fig.update_yaxes(row=z, col=1, range=[0, 1], visible=False, showticklabels=False)

    if len(to_plot)==4:
        yaxis_title_y = 0.3
    elif len(to_plot)==3:
        yaxis_title_y = 0.27
    elif len(to_plot)==3:
        yaxis_title_y = 0.25
    else:
        yaxis_title_y = 0.22

    yaxis_title_y = 0.515
        
    shared_yaxis_title = {
        'text': "Supervised                                                                                Weakly Supervised             Ground True",  # Update with your desired title
        'showarrow': False,
        'xref': 'paper',
        'yref': 'paper',
        'x': -0.1,
        'y':yaxis_title_y,
        'xanchor': 'center',
        'yanchor': 'middle',
        'textangle': -90,  # Rotate the text for vertical alignment
        'font': {'size': 15}
    }

    if 'annotations' in fig.layout:
        fig.layout.annotations += (shared_yaxis_title,)
    else:
        fig.update_layout(annotations=[shared_yaxis_title])

    return fig


# def plot_nilm_performance_comparaison(df, dataset, appliance, metric):
#     df_case = df.loc[df['Case']==appliance].copy()
    
#     order = [f'{p}DataForTrain' for p in [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 'AllPossible']] + ['All']
#     df_case['OrderPercDataTrain'] = pd.Categorical(df_case['PercDataTrain'], categories=order, ordered=True)
#     df_case = df_case.sort_values(['Model', 'OrderPercDataTrain'])
    

#     fig = px.scatter(df_case, 
#                     x='NLabelTrain',
#                     log_x=True,
#                     y=metric, 
#                     color='Model', 
#                     symbol='Model', 
#                     title=f'{dict_measure_to_display[metric]} vs number of labels used for training by each Model.', 
#                     labels={'Metric': dict_measure_to_display[metric], 'NLabelTrain': 'Number of Labels used for Training'},
#                     hover_data=['Model', 'TrainingTime'])
#     fig.update_traces(mode='markers+lines')

#     return fig


def plot_nilm_performance_comparaison(df, dataset, appliance, metric):
    df_case = df.loc[df['Case'] == appliance].copy()
    
    # Define the order for categorical data
    order = [f'{p}DataForTrain' for p in [0.025, 0.05, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'AllPossible']] + ['All']
    df_case['OrderPercDataTrain'] = pd.Categorical(df_case['PercDataTrain'], categories=order, ordered=True)
    df_case = df_case.sort_values(['Model', 'OrderPercDataTrain'])
    df_case['TrainingTime'] = df_case['TrainingTime'].round(2)
    df_case['TrainingTimeScaled'] = 10 + (df_case['TrainingTime'] / df_case['TrainingTime'].max()) * 20

    # Create a scatter plot using Plotly graph_objects
    fig = go.Figure()

    # Group data by 'Model'
    grouped = df_case.groupby('Model')

    for model, group in grouped:
        fig.add_trace(go.Scatter(
            x=group['NLabelTrain'],
            y=group[metric],
            customdata=group[[metric, 'NLabelTrain', 'TrainingTime']].round(3),
            mode='markers+lines',
            legendgroup='Weakly Supervised' if (model=='CRNN (Weak)' or model=='CamAL') else 'Supervised',  # this can be any string, not just "group"
            legendgrouptitle_text='Weakly Supervised' if (model=='CRNN (Weak)' or model=='CamAL') else 'Supervised',
            marker=dict(size=group['TrainingTimeScaled'], opacity=0.8, color=dict_color_model[model]),
            name=model,
            line=dict(width=1, color=dict_color_model[model]),
            text=group['TrainingTime'],
            hovertemplate='<br>'.join([f'{dict_measure_to_display[metric]}:'+' %{customdata[0]}', 
                                       f'Number of label:'+' %{customdata[1]}', 
                                       'TrainingTime: %{customdata[2]} Seconds'])
            )
        )

    # Update layout settings
    fig.update_layout(
        title=f'{dict_measure_to_display[metric]} vs number of labels used for training by each approach',
        xaxis=dict(
            title='Number of Labels used for Training',
            type='log'
        ),
        yaxis=dict(
            title=dict_measure_to_display[metric]
        ),
    )

    return fig


def plot_nilm_performance_comparaison_trainingtime_vs_accuracy(df, dataset, appliance, metric):
    df_case = df.loc[df['Case'] == appliance].copy()
    
    # Define the order for categorical data
    order = [f'{p}DataForTrain' for p in [0.025, 0.05, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'AllPossible']] + ['All']
    df_case['OrderPercDataTrain'] = pd.Categorical(df_case['PercDataTrain'], categories=order, ordered=True)
    df_case = df_case.sort_values(['Model', 'OrderPercDataTrain'])
    df_case['TrainingTime'] = df_case['TrainingTime'].round(2)
    #df_case['TrainingTimeScaled'] = 10 + (df_case['TrainingTime'] / df_case['TrainingTime'].max()) * 20
    df_case[metric+'_scaled'] = 10 + (df_case[metric] / df_case[metric].max()) * 20

    # Create a scatter plot using Plotly graph_objects
    fig = go.Figure()

    # Group data by 'Model'
    grouped = df_case.groupby('Model')

    for model, group in grouped:
        fig.add_trace(go.Scatter(
            x=group['NLabelTrain'],
            y=group['TrainingTime'],
            customdata=group[[metric, 'NLabelTrain', 'TrainingTime']].round(3),
            mode='markers+lines',
            legendgroup='Weakly Supervised' if (model=='CRNN (Weak)' or model=='CamAL') else 'Supervised',  # this can be any string, not just "group"
            legendgrouptitle_text='Weakly Supervised' if (model=='CRNN (Weak)' or model=='CamAL') else 'Supervised',
            marker=dict(size=group[metric], opacity=0.8, color=dict_color_model[model]),
            name=model,
            line=dict(width=1, color=dict_color_model[model]),
            text=group['TrainingTime'],
            hovertemplate='<br>'.join([f'{dict_measure_to_display[metric]}:'+' %{customdata[0]}', 
                                       f'Number of label:'+' %{customdata[1]}', 
                                       'TrainingTime: %{customdata[2]} Seconds'])
            )
        )

    # Update layout settings
    fig.update_layout(
        title=f'{dict_measure_to_display[metric]} vs number of labels used for training by each approach',
        xaxis=dict(
            title='Number of Labels used for Training',
            type='log'
        ),
        yaxis=dict(
            title=dict_measure_to_display[metric]
        ),
    )

    return fig

def plot_nilm_performance_comparaison_(df, dataset, appliance, metric):
    df_case = df.loc[df['Case'] == appliance].copy()
    
    # Define the order for categorical data
    order = [f'{p}DataForTrain' for p in [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'AllPossible']] + ['All']
    df_case['OrderPercDataTrain'] = pd.Categorical(df_case['PercDataTrain'], categories=order, ordered=True)
    df_case = df_case.sort_values(['Model', 'OrderPercDataTrain'])
    df_case['TrainingTime'] = df_case['TrainingTime'].round(2)
    df_case['TrainingTimeScaled'] = 10 + (df_case['TrainingTime'] / df_case['TrainingTime'].max()) * 20

    # Create a scatter plot using Plotly graph_objects
    fig = go.Figure()

    # Group data by 'Model'
    grouped = df_case.groupby('Model')

    #dict_color_model = {'BiGRU': 'grey', 'TPNILM': 'wheat', 'TransNILM': 'lightgreen', 'UNet-NILM': 'powderblue', 'CamAL': 'indianred', 'CRNN (Weak)': 'peachpuff', 'CRNN': 'lightgrey'}

    dict_color_model = {
    'BiGRU': 'grey', 
    'TPNILM': 'sandybrown', 
    'TransNILM': 'mediumseagreen', 
    'UNet-NILM': 'powderblue', 
    'CamAL': 'indianred', 
    'CRNN (Weak)': 'peachpuff', 
    'CRNN': 'lightgrey'
}

    for model, group in grouped:
        fig.add_trace(go.Scatter(
            x=group['NLabelTrain'],
            y=group[metric],
            customdata=group[[metric, 'NLabelTrain', 'TrainingTime']].round(3),
            legendgroup='Weakly Supervised' if (model=='CRNN (Weak)' or model=='CamAL') else 'Supervised',  # this can be any string, not just "group"
            legendgrouptitle_text='Weakly Supervised' if (model=='CRNN (Weak)' or model=='CamAL') else 'Supervised',
            mode='markers+lines',
            marker=dict(size=20, color=dict_color_model[model]), # Assign color using color dictionary
            line=dict(width=1, color=dict_color_model[model]), # Assign line color using color dictionary
            name=model,
        ))

    # Update layout settings
    fig.update_layout(
        title=f'{metric} vs number of labels used for training by each approach',
        xaxis=dict(
            title='Number of Labels used for Training',
            type='log'
        ),
        yaxis=dict(
            title=metric
        ),
    )

    return fig




def plot_detection_probabilities(data):
    # Determine the number of appliances to plot
    num_appliances = len(data)
    appliances = list(data.keys())

    # Create subplots: one row, as many columns as there are appliances
    fig = make_subplots(rows=1, cols=num_appliances, subplot_titles=appliances, shared_yaxes=True)

    for i, appliance in enumerate(appliances, start=1):
        appliance_data = data[appliance]
        models = ['CamAL']
        probabilites = [appliance_data['pred_prob']]
        #color_model  = [dict_color_model['NILMCAM']]

        # Add bars for each class in the subplot
        fig.add_trace(go.Bar(x=models, y=probabilites,  marker_color='peachpuff'), row=1, col=i)

    for axis in fig.layout:
        if axis.startswith('yaxis'):
            fig.layout[axis].update(
                range=[-0.1, 1.1],
                tickmode='array',
                tickvals=[0, 0.5, 1],
                ticktext=['Not Detected', '0.5', 'Detected']
            )

    # Update layout once, outside the loop
    fig.update_layout(
        title_text='Appliance(s) detection probabilities',
        barmode='group',
        showlegend=False,
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1, # gap between bars of the same location coordinate.
        height=400, # You can adjust the height based on your needs
        width=1000, # Adjust the width based on the number of appliances or your display requirements
    )

    return fig



def plot_signatures(appliances):
    fig = make_subplots(rows=1, cols=len(appliances), subplot_titles=[f'{appliance}' for appliance in appliances], shared_yaxes=True)

    for i, appliance in enumerate(appliances, start=1):
        signature = pd.read_csv(os.getcwd()+f'/Data/example_{appliance}.gzip', parse_dates=['Time'], compression='gzip').set_index('Time')
        signature = signature.resample('1min').mean()

        fig.add_trace(go.Scatter(x=signature.index, y=signature[appliance], 
                                 marker_color=dict_color_appliance[appliance], 
                                 mode='lines', fill='tozeroy'),
                          row=1, col=i)
        
      # Update y-axes for each subplot to have the range [0, 1]
    for j in range(1, len(appliances) + 1):
        fig.update_xaxes(title_text="Time", row=1, col=j)
        
    fig.update_layout(title='Example of signature for different appliances', 
                      yaxis_title='Power (Watts)', 
                      showlegend=False,
                      height=300, 
                      margin=dict(l=100, r=30, t=70, b=40),
                      yaxis_range=[0, 10000 if 'Shower' in appliances else 6000]
                    )

    return fig
