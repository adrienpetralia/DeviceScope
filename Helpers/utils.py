import streamlit as st

# === Lib import === #
import os, lzma, io, pickle
import numpy as np
import pandas as pd
import torch

# === Customs import === #
#from Utils.constants import *

from Models.Classifiers.ResNet3 import ResNet3
from Helpers.class_activation_map import CAM


def sigmoid(z):
    return 2 * (1.0/(1.0 + np.exp(-z))) - 1


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

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
    # elif resnet_name =='ResNet3LN':
    #     inst = ResNet3LN(kernel_sizes=[kernel_size, 7, 3], **kwargs)
    # elif resnet_name=='ResNet5':
    #     inst = ResNet5(kernel_sizes=[kernel_size, 7, 3], **kwargs)
    # elif resnet_name =='ResNet5LN':
    #     inst = ResNet5LN(kernel_sizes=[kernel_size, 7, 3], **kwargs)
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



def get_pred_nilmcam_one_appliance(dataset_name, window_agg, appliance):

    path_ensemble = os.getcwd()+f'/TrainedModels/{dataset_name}/1min/{appliance}/ResNetEnsemble/'
    pred_prob, soft_label, soft_label_before_sig, avg_cam = get_soft_label_ensemble(window_agg, path_ensemble)

    if appliance=='Microwave':
        soft_label = sigmoid(np.array(soft_label_before_sig))
        soft_label = np.where(soft_label > 0.2, 1, 0)

    elif appliance=='WashingMachine':
        soft_label = sigmoid(np.array(soft_label_before_sig))
        soft_label = np.where(soft_label > 0.2, 1, 0)

    return {'pred_prob': pred_prob, 'pred_status': soft_label, 'soft_label_before_sig': soft_label_before_sig, 'avg_cam': avg_cam}


def pred_one_window_nilmcam(k, df, window_size, dataset_name, appliances):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    window_agg = window_df['Aggregate']

    pred_dict = {}
    for appl in appliances:
        pred_dict[appl]  = get_pred_nilmcam_one_appliance(dataset_name, window_agg, appl)

    return pred_dict
