
# Wavelet Probabilistic Neural Networks

This repo contains source codes for the paper **Wavelet Probabilistic Neural Networks (WPNN)** published in IEEE Transactions on Neural Networks and Learning Systems.

Link to the paper: TBD

  

## Software Requirement

* MATLAB 2014a

  

## Main Experiments

* Online stationary experiment (./online stationary/test_wpnn_online_stationary.m): 
	* Main function:
		fun_WPNN_pdf_online_stationary_prequential.m - function to update the network parameter in the stationary environment. 
	* Run "./online stationary/test_wpnn_online_stationary.m"

* Online non-stationary (./online nonstationary/test_wpnn_online_non_stationary.m): 
	* Main function:
	fun_WPNN_pdf_online_nonstationary_prequential.m - function to update the network parameter in the non-stationary environment. 
	* Run *./online nonstationary/test_wpnn_online_non_stationary.m* & *test_wpnn_online_non_stationary_nsl_kdd.m* for synthetic and real-world datasets.

* Options:
	* Modify the hyper-parameters accordingly in the main funcitons: 
		* the order B-spline, 
		* <img src="https://render.githubusercontent.com/render/math?math=j_0">
		* <img src="https://render.githubusercontent.com/render/math?math=\alpha"> 
	* Replace your own datasets:
		* load('./datasets/_YOUR_OWN_DATASET_')

## File Directory
```
.
├── Online Stationary Experiment/
│   ├── WPNN/
│   │   └── test_wpnn_online_stationary.m %main function for WPNN online stationary environment/
│   │       ├── fun_radial_bspline.m
│   │       ├── fun_relevant_frame_for_given_datapoint.m
│   │       ├── fun_wpnn_evaluation_dual_class.m
│   │       ├── fun_wpnn_initialisation.m
│   │       ├── fun_wpnn_online_updating_stationary.m %Eq:11
│   │       └── fun_WPNN_pdf_online_stationary_prequential.m
│   ├── KDE/
│   │   └── test_pnn_online_stationary.m %main function for KDE-based PNN/
│   │       ├── fun_kde_testing_online_latest_pt.m
│   │       ├── fun_kde_testing_online_window.m
│   │       ├── funmyKDE.m
│   │       └── test_pnn_online_stationary.m
│   └── datasets/
│       └── online_stationary_datasets.mat %synthetic dataset
└── Online Non-Stationary Experiment/
    ├── WPNN/
    │   └── test_wpnn_online_stationary.m %main function for WPNN online stationary environment/
    │       ├── fun_radial_bspline.m
    │       ├── fun_relevant_frame_for_given_datapoint.m
    │       ├── fun_wpnn_evaluation_dual_class.m
    │       ├── fun_wpnn_initialisation.m
    │       ├── fun_wpnn_online_updating_nonstationary.m %Eq:12
    │       └── fun_WPNN_pdf_online_stationary_prequential.m
    ├── KDE/
    │   └── test_pnn_online_stationary.m %main function for KDE-based PNN/
    │       ├── fun_kde_testing_online_latest_pt.m
    │       ├── fun_kde_testing_online_window.m
    │       ├── funmyKDE.m
    │       └── test_pnn_online_stationary.m
    └── datasetss
        ├── non_stationary_dataset.mat.mat
        ├── http.mat 
        └── NSL_KDD.mat

```

Authors: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria

Title: Wavelet Probabilistic Neural Networks

IEEE Transactions on Neural Networks and Learning Systems

2022

Institution: Imperial College London

Date: Jan-2022
