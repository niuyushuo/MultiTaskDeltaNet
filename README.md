<!-- [![DOI](https://zenodo.org/badge/1024057265.svg)](https://doi.org/10.5281/zenodo.16415664)-->
[![DOI](https://zenodo.org/badge/1024057265.svg)](https://zenodo.org/badge/latestdoi/1024057265)
# MultiTaskDeltaNet (MTDN)
Change Detection-based Image Segmentation for operando ETEM with Application to Carbon Gasification Kinetics
## Network
<img src="https://github.com/niuyushuo/MultiTaskDeltaNet/blob/7f950176eab5de6c4512032d140f1fda05d82265/Results/image_github/Model_arch.png" width="500" height="400">

## Installation
Create a conda environment:
```
conda create -n python3.10_pytorch2.0 python=3.10
conda activate python3.10_pytorch2.0
```

Install pytorch based on your cuda version:
```
nvidia-smi
```

<img src='https://github.com/niuyushuo/MultiTaskDeltaNet/blob/cf35d0590df76a88bdd18f4d15cc7f719b90e6bd/Results/image_github/smi.png' width="500" height="400">

<img src='https://github.com/niuyushuo/MultiTaskDeltaNet/blob/cf35d0590df76a88bdd18f4d15cc7f719b90e6bd/Results/image_github/pytorch.png' width="400" height="200">

Install pytorch:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install the remaining packages:
```
conda install matplotlib
conda install esri::einops
conda install anaconda::pandas
conda install anaconda::scikit-learn
conda install anaconda::seaborn
conda install anaconda::openpyxl
pip install "ray[tune]"  #### if hyper-parameter tuning needed

```

## Dataset
The dataset file contains a separate folder for each video ID.

To help you better understand the dataset, here are a few key points:

1. **Video IDs:** The available video IDs are 101, 102, 103, 201, 203, 301, and 302. There is a relationship between video IDs and filament IDs mentioned in the paper. The training dataset consists of video IDs 102_R1, 102_R2, and 302, which correspond to filament IDs 1, 2, and 3. The validation dataset includes video IDs 103 and 301, related to filament IDs 4 and 5. The test dataset contains video IDs 201 and 203, corresponding to filament IDs 6 and 7. For quality reasons, dataset 101 has not been used.

2. **Time:** There is a relationship between Frame and Time. The image names and model prediction results are saved using Frame numbers. However, in the paper, we report the results using Time numbers. For instance, Frame 88 corresponds to Time 0. For detailed relationships, see the "performance_F1_individual.xlsx" file in the Results "folder". 

3. **Folder Structure:** Each video ID folder contains two subfolders: "original" and "results." 
   - The "original" folder holds all the original images without cropping. Within this folder, the "area1" subfolder contains images labeled A1, the "area2" subfolder contains images labeled A2, and the "img" subfolder has the original filament images. ‘_background_patch.png’ is a pixel used to mask unselected filamentous carbon in the cropped image.
   - The "results" folder contains images after preprocessing, which will have filament images without overlapping structures. Similarly, the "area1," "area2," and "area1-2" subfolders contain the A1 labels, A2 labels, and the difference between A1 and A2 (A1 - A2), respectively. The "img" subfolder in the results folder also contains filament images. It’s important to note that the 102 folder is a special case; it only contains the original images, while the result images are located in the 102_R1 and 102_R2 folders.

4. **Testing the Code:** To test the code, please download the entire dataset, as the code is designed to read the dataset based on this specific structure.


## Code
The code folder contains all scripts required to train the models, perform hyperparameter search, and generate predictions. It also includes a checkpoints folder with saved models and a vis folder that holds raw prediction outputs for each model.

To help you better understand and use the code, here are a few key points:

1. **Model Training:**
   
  - MTDN Model: We have separate code for the MTDN and U-net models. For the MTDN model, the file `main_siam_train.py` allows you to train various versions of MTDN. You can select the `trainer_siam` option for MTDN based on whether you want no_init, init1, or init2. If you wish to train any of these versions, make sure to select and comment out the relevant lines in `trainer_siam.py`. This file contains detailed instructions on how to proceed. If you want to assess model performance immediately after training, you will also need to modify `test_siam_cd.py` accordingly. For the `trainer_siam_ablation`, this trains a single-task model. To do this, you need to train MTDN for A1 and A2 separately, which again requires selecting and commenting out the appropriate lines in `trainer_siam_ablation.py`. Each of these files includes comments to assist you in running the code and training the model.
   
  - U-net Model: For the U-net model, the process is similar to training the MTDN single-task model. You will need to select and comment on the relevant lines in `trainer_unet.py` for either A1 or A2.

2. **Model Testing:**

For testing pre-trained models, use pred_siam.py (for all MTDN variants) and pred_unet.py (for U-net). Again, select the model variant by modifying the appropriate lines.

3. **Hyperparameter Tuning:**
 - The default training scripts include fixed seeds and recommended hyperparameters to reproduce results in the paper.
 - To run your own hyperparameter search, use main_siam_raytune.py for MTDN or main_unet_raytune.py for U-net. Remember to comment/uncomment lines to choose the desired model variant.

4. **Model Outputs:**

 - Trained models are saved in the checkpoints folder.

 - Prediction plots and performance summaries are saved in the vis folder. Additional evaluation results can be found in the Results folder.

5. **Path Configuration:**

 - Before running any code, update the paths in the scripts to match the location of your dataset, model checkpoints, and output directories.


## Results
This section includes both the numerical results and visual predictions for each model. The overall performance is saved in the "performance_F1_individual.xlsx" file. Results are organized by model type, validation/test dataset, and prediction category.

1. **Folder Structure:**
Each model folder (no_init, init1, init2 and U-net) contains results for both the validation and test datasets (103, 201, 203,301). Inside each dataset subfolder, you will find:

 - 'area1' and 'area2': Raw model prediction masks for A1 and A2.

 - 'img': Original filament images used for inference.

 - 'output': Combined A1 + A2 visualizations, as presented in the paper.

 - 'F1_A1.txt' and 'F1_A2.txt': F1 scores for each sample, saved as plain text.

 - 'groundtruth': Ground truth masks for A1 and A2 (available for both validation and test sets).

2. **Confusion Matrix Visualizations:**
 - The Comparison/ folder includes confusion matrix plots shown in Figures 6 and 7 of the paper.

 - These plots are provided for MTDN_init2 and U-Net, as they are the primary models discussed in the paper.

 - If you would like to generate confusion matrices for other variants (e.g., no_init, init1, etc.), use:

   - A1_comparison.ipynb

   - A2_comparison.ipynb

3. **Images for README:**
 - The image_github/ folder includes curated visualizations used in this README.md file.

## Citation
If you use **MultiTaskDeltaNet** in your research, please cite:

**APA:**
> Niu, Y., Li, T., Zhu, Y., & Yang, Q. (2025). *MultiTaskDeltaNet: Change Detection-based Image Segmentation for operando ETEM with Application to Carbon Gasification Kinetics*. Digital Discovery. [https://doi.org/10.1039/D5DD00333D](https://doi.org/10.1039/D5DD00333D)

**BibTeX:**
```bibtex
@article{niu2025multitaskdeltanet,
  title      = {MultiTaskDeltaNet: Change Detection-based Image Segmentation for operando ETEM with Application to Carbon Gasification Kinetics},
  author     = {Niu, Yushuo and Li, Tianyu and Zhu, Yuanyuan and Yang, Qian},
  journal    = {Digital Discovery},
  year       = {2025},
  publisher  = {Royal Society of Chemistry}
}
```

## License
This project is licensed under the terms of the [MIT License](LICENSE).

**Yushuo Niu**  
Ph.D. Candidate, University of Connecticut  
yushuo.niu@uconn.edu
