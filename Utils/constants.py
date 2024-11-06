# ========================== Texts ========================== #

text_tab_playground = f"""
# Playground
Attempt to identify the appliances in the aggregate power by looking at the provided example patterns at the bottom of the frame or select the ones you want automatically detect using DeviceScope!"""

text_tab_benchmark = f"""
# Benchmark
Explore and compare the performance of our method. 

First select a **Dataset**, a **Detection Metric**, to measure *if* an appliance is well detected in a given window, and a **Localization Metric**, to measure *when* the appliance is well localize.
"""


text_description_dataset  = f"""
The electricity consumption data available to test our system come from two different studies and are available publicy online.
Each dataset is composed of several houses that have been monitored by sensors that record the total main power and appliance-level power for a period of time.

- [UKDALE](https://jack-kelly.com/data/): The UK-DALE dataset contains data from 5 houses in the United Kingdom and includes appliance-level and aggregate load curves sampled at a minimum sampling rate of 6 seconds.
Four houses were recorded for over a year and a half, while the 5th was recorded for 655 days.

- [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned): The REFIT dataset contains data from 20 different houses in the United Kingdom that have been monitored with smart meters and multiple sensors. 
This dataset provides aggregate and individual appliance load curves at 8-second sampling intervals. 

- [IDEAL](https://www.nature.com/articles/s41597-021-00921-y): The IDEAL dataset contains data from 255 different houses in the United Kingdom that have been monitored with smart meters.
For a subgroup of 39, submeters data are available for different appliances.
"""

text_description_model  = f"""
Appliance detection can be cast as a time series classification problem, where a classifier is trained in a binary supervised manner to detect the presence of an appliance in a consumption series. 
Following results from previous studies, it has been proven that deep learning-based methods are the most accurate and efficient for tackling this task.
Our system provide 4 different time series classifiers to detect appliance:

- ConvNet: Convolutional Neural Network (CNN) is a well-knwon deep learning architecture commonly used in image recognition. 
In this system, we use a variant designed for time series classification. 
The architecture employs stacked 1D convolutional blocks with specific kernel sizes and filters, followed by global average pooling and linear layers for classification.


- ResNet: The Residual Network (ResNet) architecture was originally proposed to adress the gradient vanishing problem in deep CNNs. 
It is also a well-knwon architecture to perform image recognition.
As for the ConvNet, we use an variant proposed for time series classification.
It consists of stacked residual blocks with residual connections, where each block contains 1D convolutional layers with the same kernel sizes and filters. 
A global average pooling, a linear layer, and a softmax activation are used for classification.


- InceptionTime: Inspired by inception-based networks for image classification, InceptionTime is designed for time series classification.
The model employs Inception modules composed of concatenated 1D convolutional layers using different filter sizes.
The outputs are passed through activation and normalization layers; at the end, classification is performed using a global average pooling, followed by a linear layer and softmax activation function.


- TransApp: A recent study proposed TransApp, a deep-learning time series classifier specially designed to detect appliances in smart meter consumption series.
The architecture results in an embedding block made of convolutional layers that serves as a features extractor followed by multiple Transformer layers.
In the end, global average pooling, followed by a linear layer and softmax activation function, performs the classification.
As the architecture was originally proposed for detecting appliances in large datasets, we adapted it for our system as a smaller and simplified version by keeping only one Transformer layer after the embedding block.
"""

text_description_explainability = r"""
Identifying the discriminative features that influence a classifier's decision-making process has been extensively studied in the literature.
For classification using deep-learning-based algorithms, different methods have been proposed to highlight the parts of an input instance that contribute the most to the final decision of the classifier.

**Class Activation Map:** Originally proposed for explaining the decision-making process of deep-learning classifier in computer vision, the Class Activation Map (CAM) enables the highlighting of the parts of an image that contributed the most to obtaining the predicted label. 
A one-dimensional adaptation of this method was proposed for time series classification to highlight the relevant subsequences of a time series.
Note that CAM is only usable with CNN-based architectures incorporating a GAP layer before the softmax classifier.
For univariate time series as load curves, the CAM for the label $c$ is defined as follows:

$$
\text{CAM}_c = \sum_k w_k \cdot f_k(t)
$$
where $w_k$ are the weights of the $k^{th}$ filter associated to class $c$, and $f_k(t)$ are the inner features at a certain a timestamp $t$.
It results in a univariate time series where each element (at the timestamp $t  \in [1, T ]$) equals the weighted sum of the data points at $t$, with the weights learned by the neural network and reflect the importance of each timestamp.
"""

text_intro_whats_behind = f"""
## Interactive detection and localization of appliance patterns in electricity consumption time series

Electricity suppliers have installed millions of smart meters worldwide to improve the management of the smart grid system.
These meters capture detailed time-stamped electricity consumption of the total main power consumed in a house: this recorded signal is hard to analyze as it regroups multiple appliance signatures that run simultaneously.
Making non-expert users (as consumers or sales advisors) understand it has become a major challenge for electricity suppliers.
We propose Deviscope as an interactive solution to facilitate the understanding of electrical data by detecting and localizing individual appliance patterns within recorded time periods.
"""

text_camal_info = f"""
### How DeviceScope works?
The core of our system is based on a combination of recent works conducted on appliance detection [[1](https://arxiv.org/abs/2305.10352), [2](https://arxiv.org/abs/2401.05381)] and explainable classification [[3](https://arxiv.org/abs/1611.06455), [4](https://helios2.mi.parisdescartes.fr/~themisp/dCAM/), [5](https://epfml.github.io/attention-cnn/)].
In a nutshell, DeviceScope uses a trained time series classifier to detect ***if*** an appliance is used in a given period of time. If this is the case, a explainable classification approach is applied to detect ***when*** the device is running. 
It works by highlighting the portion of the input consumption series that had the most significant impact on the classifier's decision.
Unlike other works conducted in this area (NILM based approaches [[6](https://helios2.mi.parisdescartes.fr/~themisp/publications/energybuildings24.pdf)]), our method is based on a weakly supervised process and therefore requires far fewer labeled data.

To learn more about the DeviceScope system, look at the two sections below!
"""


text_info = f"""
### Contributors

* [Adrien Petralia](https://adrienpetralia.github.io/), EDF R&D, Université Paris Cité
* [Paul Boniol](https://boniolp.github.io/), Inria, ENS, PSL University, CNRS
* [Philippe Charpentier](https://www.researchgate.net/profile/Philippe-Charpentier), EDF R&D
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/), Université Paris Cité, IUF

### Acknowledgments
Work supported by EDF R&D and ANRT French program.
"""

# ========================== Colors ========================== #
dict_color_appliance = {'WashingMachine': 'teal', 'Dishwasher': 'skyblue', 'Shower': 'brown', 'Kettle': 'orange', 'Microwave': 'grey'}
#dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransAppS': 'indianred', 'Ensemble': 'peachpuff', 'CamAL': 'peachpuff'}
dict_measure_to_display = { 'ACCURACY': 'Accuracy', 'BALANCED_ACCURACY': 'Balanced Accuracy', 'F1_SCORE': 'F1 Score', 'RECALL': 'Recall', 'PRECISION': 'Precision'}


# ========================== Lists ========================== #
lengths_list   = ['6 hours', '12 hours', '1 Day']

devices_list_refit_ukdale = ['Kettle', 'Dishwasher', 'WashingMachine', 'Microwave']
devices_list_ideal        = ['Dishwasher', 'Shower', 'WashingMachine']

measures_list  = ['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Precision', 'Recall']

list_dataset   = ['UKDALE', 'REFIT', 'IDEAL']

list_ukdale_ts = ['UKDALE_House2_2013-05', 'UKDALE_House2_2013-06', 'UKDALE_House2_2013-07', 'UKDALE_House2_2013-08', 'UKDALE_House2_2013-09', 'UKDALE_House2_2013-10']

list_ideal_ts  = ['IDEAL_House65_2017-06', 'IDEAL_House65_2018-02', 'IDEAL_House65_2018-04', 'IDEAL_House175_2018-01']

list_refit_ts  = ['REFIT_House2_2013-10', 'REFIT_House2_2013-11', 'REFIT_House2_2013-12', 'REFIT_House2_2014-01', 'REFIT_House2_2014-02', 'REFIT_House2_2014-03', 'REFIT_House2_2014-04', 'REFIT_House2_2014-05',
                  'REFIT_House2_2014-06', 'REFIT_House2_2014-07', 'REFIT_House2_2014-08', 'REFIT_House2_2014-09', 'REFIT_House2_2014-10', 'REFIT_House2_2014-11', 'REFIT_House2_2014-12', 'REFIT_House2_2015-01',
                  'REFIT_House20_2014-03', 'REFIT_House20_2014-04', 'REFIT_House20_2014-05', 'REFIT_House20_2014-06', 'REFIT_House20_2014-07', 'REFIT_House20_2014-08', 'REFIT_House20_2014-09', 'REFIT_House20_2014-10',
                  'REFIT_House20_2014-11', 'REFIT_House20_2014-12', 'REFIT_House20_2015-01', 'REFIT_House20_2015-02', 'REFIT_House20_2015-03', 'REFIT_House20_2015-04', 'REFIT_House20_2015-05', 'REFIT_House20_2015-06']
