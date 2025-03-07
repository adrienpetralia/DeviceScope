# ========================== Texts Pages explaination (What's behind) ========================== #


text_intro_whats_behind = f"""
## DeviceScope: Interactive detection and localization of appliance patterns in electricity consumption time series

Electricity suppliers have installed millions of smart meters worldwide to improve the management of the smart grid system.
These meters capture detailed time-stamped electricity consumption of the total main power consumed in a house: this recorded signal is hard to analyze as it regroups multiple appliance signatures that run simultaneously.
Making non-expert users (as consumers or sales advisors) understand it has become a major challenge for electricity suppliers.
We propose DeviceScope as an interactive tool designed to facilitate the understanding of smart meter data by detecting and localizing individual appliance patterns within a given time period.
Our system is based on **CamAL** *(**C**lass **A**ctivation **M**ap based **A**ppliance **L**ocalization)*, a novel weakly supervised approach for appliance localization that only requires the knowledge of the existence of an appliance in the household to be trained. 
"""








text_description_explainability = r"""
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

text_description_resnets = r"""
The original Residual Network architectures introduced for TSC consists of stacked residual blocks with residual connections, where each block contains 1D convolutional layers with the same kernel sizes.
At the network's end, a global average pooling (GAP) is followed by a linear layer to perform classification.
We leverage this baseline to an ensemble of 5 networks that each use different kernel size in the layers.
More precisely, we trained multiple networks using different kernels size $k$ ($k \in \{5, 7, 9, 15\}$) and keep the ones that perform the best to detect a specific appliance.
The intuition lies in the fact the receptive fields of a CNN vary according to the kernel size and therefore can produce a different explainability.
"""


text_camal_info1 = f"""
### How DeviceScope works?
**CamAL** is based on a combination of recent works conducted on appliance detection [[1](https://arxiv.org/abs/2305.10352), [2](https://arxiv.org/abs/2401.05381)] and explainable classification [[3](https://arxiv.org/abs/1611.06455), [4](https://helios2.mi.parisdescartes.fr/~themisp/dCAM/), [5](https://epfml.github.io/attention-cnn/)].
is composed of two parts: 

- An ensemble of deep-learning classifiers that performs the detection
- An explainability-based module that enables the localization of the appliance if detected.

Unlike other works conducted in this area (NILM based approaches [[6](https://helios2.mi.parisdescartes.fr/~themisp/publications/energybuildings24.pdf)]), our method is based on a weakly supervised process and therefore requires **far fewer labeled data**.
"""

text_camal_info2 = f"""
#### Step 1: Appliance Detection
Detecting if an appliance has been used in a period of time can be cast as a time series classification (TSC) problem.
To do so, a classifier is trained in a binary supervised manner to detect the presence of an appliance using only one label (0 or 1) for an entire series.
Based on the results of previous studies, deep learning-based (specifically convolutional-based), are the most efficient and accurate solutions for tackling this task.
Thus, our system is based on an ensemble of convolutional residual networks (ResNets) to detect if an appliance pattern is present in a consumption series.
"""

text_camal_info3 = r"""
Identifying the discriminative features that influence a classifier's decision-making process has been extensively studied in the literature. 
For classification using deep-learning-based algorithms, different methods have been proposed to highlight (i.e., localize) the parts of an input instance that contribute the most to the final decision of the classifier. 
Based on this previous work, we developed CamAL, a method specifically designed to localize appliance patterns in electricity consumption series.

1. **Ensemble Prediction**: The aggregated input sequence $\mathbf{x}$ is fed into an ensemble of ResNet models. Each model predicts the probability of the target appliance being active in the current window. The ensemble prediction probability is computed as the average of individual model probabilities:

   $\text{Prob}_{\text{ens}} = \frac{1}{N} \sum_{n=1}^N \text{Prob}_n,$

   where $N$ is the number of models in the ensemble, and $\text{Prob}_n$ is the prediction from the $n$-th model.

2. **Appliance Detection**: If the ensemble probability exceeds a threshold (e.g., $\text{Prob}_{\text{ens}} > 0.5$), the appliance is considered detected in the current window.

3. **CAM Extraction**: We extract each model's CAM for class 1 (appliance detected). For univariate time series, the CAM for class $c$ at timestamp $t$ is defined as:

   $$
   \text{CAM}_c(t) = \sum_{k} w_k^c \cdot f_k(t),
   $$

   where $w_k^c$ are the weights associated with the $k$-th filter for class $c$, and $f_k(t)$ is the activation of the $k$-th feature map at time $t$.

4. **CAM Processing**: Each $\text{CAM}_n$ is first normalized to the range $[0, 1]$, and we then take the average of each extracted CAM for the ensemble:

   $$
   \text{CAM}_{\text{avg}}(t) = \frac{1}{N} \sum_{n=1}^N \widetilde{\text{CAM}}_n(t).
   $$

5. **Attention Mechanism**: $\text{CAM}_{\text{avg}}$ serves as an attention mask, highlighting the ensemble decision for each timestamp. We apply this mask to the input sequence through point-wise multiplication and pass the results through a sigmoid activation function to map the values between 0 and 1:

   $$
   \mathbf{s}(t) = \text{Sigmoid}(\text{CAM}_{\text{avg}}(t) \circ \mathbf{x}(t)).
   $$

6. **Appliance Status**: The obtained signal is then rounded to obtain binary labels ($1$ if $s(t) \geq 0.5$), indicating the appliance's status at each timestamp. This results in a binary time series $\hat{y}(t)$ that represents the predicted status of the appliance.
"""



text_description_model  = f"""
Non-Intrusive Load Monitoring (NILM) refers to the challenge of estimating the power consumption, pattern, or on/off state activation of individual appliances using only the main power reading.
Early NILM solutions involved Combinatorial Optimization (CO) to estimate the proportion of total power consumption used by distinct active appliances at each time step.
NILM regained popularity in the late 2010s, following the release of smart meter datasets and recent methods proposed to tackle this task are based on supervised approaches that requires the individual appliance power to be train.
However, ground-truth labels are expensive to collect and require the installation of sensors on every appliance in the house.
In practice, the available information is merely the activation (or not) of an appliance within a time frame. 
We note that NILM approaches simply cannot operate with such scarce labels: trying to train a NILM solution with only one label for the entire series (e.g., by replicating the label for all time steps) implies that it can no longer be used to localize an appliance; indeed, NILM solutions provide a probability of detection for each individual timestamp to be able to localize it.
"""


text_tpnilm  = f"""
- [TPNILM](https://www.mdpi.com/2076-3417/10/4/1454): 
The authors proposed to address the problem of appliance state recognition using a fully convolutional deep neural network, borrowing some techniques used in the semantic segmentation of images and multilabel classification.
The archietctures consist of several convolutional layers using pooling.
"""

text_unetnilm  = f"""
- [Unet-NILM](hhttps://dl.acm.org/doi/10.1145/3427771.3427859): 
"""

text_transnilm  = f"""
- [TransNILM](https://ieeexplore.ieee.org/document/9991439): 
"""

text_bigru = f"""
- [BiGRU](https://link.springer.com/article/10.1007/s11227-023-05149-8): 
"""

text_crnn = f"""
- [CRNN](https://ieeexplore.ieee.org/document/9831435): 
"""




text_description_dataset  = f"""
The electricity consumption data available to test our system come from two different studies and are available publicy online.
Each dataset is composed of several houses that have been monitored by sensors that record the total main power and appliance-level power for a period of time.

- [IDEAL](https://www.nature.com/articles/s41597-021-00921-y): The IDEAL dataset contains data from 255 different houses in the United Kingdom that have been monitored with smart meters.
For all of the households, survey information are available, such as the type of appliances present in the households.
For a subgroup of 39, submeters data are available for different appliances.

- [UKDALE](https://jack-kelly.com/data/): The UK-DALE dataset contains data from 5 houses in the United Kingdom and includes appliance-level and aggregate load curves sampled at a minimum sampling rate of 6 seconds.
Four houses were recorded for over a year and a half, while the 5th was recorded for 655 days.

- [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned): The REFIT dataset contains data from 20 different houses in the United Kingdom that have been monitored with smart meters and multiple sensors. 
This dataset provides aggregate and individual appliance load curves at 8-second sampling intervals. 

"""
