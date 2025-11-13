# SHSeg

This dataset contains masks for athletes that are currently skiing. It has been published alongside the following paper: 

    @misc{schön2025skipclickcombiningquickresponses,
        title={SkipClick: Combining Quick Responses and Low-Level Features for Interactive Segmentation in Winter Sports Contexts},
        author={Robin Schön and Julian Lorenz and Daniel Kienzle and Rainer Lienhart},
        year={2025},
        eprint={2501.07960},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2501.07960},
    }

 

If you intend to use our dataset in your publication, please remember to cite our paper.

The dataset on its own only contains masks. The corresponding images are a subset of the SkiTB dataset published alongside the following two publications: 

 

    @InProceedings{SkiTBwacv,
        author = {Dunnhofer, Matteo and Sordi, Luca and Martinel, Niki and Micheloni, Christian},
        title = {Tracking Skiers from the Top to the Bottom},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
        month = {Jan},
        year = {2024}
    }


    @article{SkiTBcviu,
        title = {Visual tracking in camera-switching outdoor sport videos: Benchmark and baselines for skiing},
        author = {Matteo Dunnhofer and Christian Micheloni}, 
        journal = {Computer Vision and Image Understanding},
        volume = {243},
        pages = {103978},
        year = {2024},
        doi = {https://doi.org/10.1016/j.cviu.2024.103978},
    }

 

If you use their images, please also cite their paper. You can download the images from [here](https://machinelearning.uniud.it/datasets/skitb/). 

 

The dataset itself can be downloaded from  [here](https://myweb.rz.uni-augsburg.de/~schoerob/datasets/shseg/SHSeg_masks_only.zip). The ZIP-file contains a file README.txt which contains further instructions on how to use the dataset.

# SkipClick
## Acknowledgements
The code in this repository is based on [SimpleClick](https://github.com/uncbiag/SimpleClick) and [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation). 
Please also cite their papers if you make use of this repository: 

    @InProceedings{Liu_2023_ICCV,
        author    = {Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
        title     = {SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2023},
        pages     = {22290-22300}
    }
    
    @inproceedings{ritm2022,
      title={Reviving iterative training with mask guidance for interactive segmentation},
      author={Sofiiuk, Konstantin and Petrov, Ilya A and Konushin, Anton},
      booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
      pages={3141--3145},
      year={2022},
      organization={IEEE}
    }
    
    @inproceedings{fbrs2020,
       title={f-brs: Rethinking backpropagating refinement for interactive segmentation},
       author={Sofiiuk, Konstantin and Petrov, Ilia and Barinova, Olga and Konushin, Anton},
       booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
       pages={8623--8632},
       year={2020}
    }


## Weights
You can download the weights from these links: 

| **Configuration** | **GrabCut NoC@90** | **Avg. WSESeg NoC@90** | **Model File** | **Link to Weights** |
|--------|--------|--------|--------|--------|
| Baseline | 1.74 | 12.031 | [Link](https://github.com/Schorob/skipclick/blob/main/models/iter_mask/featurevit_base448_cocolvis_itermask_unfrozen.py) | [Link](https://mediastore.rz.uni-augsburg.de/get/Gf3Aza1fSE/) |
| + Frozen Backbone | 1.72 | 11.951 | [Link](https://github.com/Schorob/skipclick/blob/main/models/iter_mask/featurevit_base448_cocolvis_itermask.py) | [Link](https://mediastore.rz.uni-augsburg.de/get/2JipuNEGYO/) |
| + Intermediate Features | 1.40 | 10.344 | [Link](https://github.com/Schorob/skipclick/blob/main/models/iter_mask/featurevit_base448_cocolvis_itermask_intermediate.py) | [Link](https://mediastore.rz.uni-augsburg.de/get/aOa4lakFcM/) |
| + Skip Connections | 1.44 | 9.163 | [Link](https://github.com/Schorob/skipclick/blob/main/models/iter_mask/featurevit_base448_cocolvis_itermask_intermediate_skip.py) | [Link](https://mediastore.rz.uni-augsburg.de/get/MQjMyFYnZ7/) |

In order to run an evaluation of any of the pretrained models, you will have to adapt the content of `config.yml`, as this file contains the paths to the datasets. 
The `scripts/evaluation.py` can be run by: 
``` 
python3 scripts/evaluate_model.py NoBRS --eval-mode=cvpr --gpus=[GPU-Number] --eval-fvit --suppress-zoom --checkpoint=[/path/to/the/checkpoint.pth] --datasets=[DatasetName]
```

## SkipClick with a DINOv3 Backbone 
We also trained a version of the full SkipClick architecture with a DINOv3-based backbone. When compared to the original DINOv2-based backbone, we obtain the following results: 

| **Configuration**  | **GrabCut NoC@90** | **Berkeley NoC@90** | **SBD NoC@90** | **DAVIS NoC@90** | **HQSeg-44k NoC@90** | **Model File** | **Link to Weights** |
|--------------------|--------------------|---------------------|----------------|------------------|----------------------|----------------|---------------------| 
| SkipClick (DINOv2) | 1.44 | 2.45 | 6.18 | 4.94 | 6.00 | [Link](https://github.com/Schorob/skipclick/blob/main/models/iter_mask/featurevit_base448_cocolvis_itermask_intermediate_skip.py) | [Link](https://mediastore.rz.uni-augsburg.de/get/MQjMyFYnZ7/) |
| SkipClick (DINOv3) | 1.42 | 2.45 | 6.16 | 4.79 | 5.77 | [Link](https://github.com/Schorob/skipclick/blob/master/models/iter_mask/featurevit_dinov3_base448_cocolvis_itermask_intermediate_skip.py) | [Link](https://mediastore.rz.uni-augsburg.de/get/EkSQanPSOL/) |

## GUI Demo 
The folder `novel_demo` contains a Qt6 based basic GUI for the usage of the SkipClick model with quick responses. 
Before running it, you will have to make same preparatory steps: 
 - Create a virtual environment using `venv`: `python -m venv skipclick_venv` (Note: We used Python 3.10.12)
 - Activate the environment
 - Update pip: `python -m pip install --upgrade pip` 
 - Install the `torch` and `torchvision` packages (see [the official guide](https://pytorch.org/get-started/locally/)). We used `torch==2.9.0`, `torchmetrics==1.8.2` and `torchvision==0.24.0`. 
 - Install the `wheel` package (prerequisite for `mmcv-full`): `python -m pip install wheel`
 - Install `mmcv-full`: `python -m pip install mmcv-full` 
 - Install all other repository requirements from `requirements.txt` in the repository: `python -m pip install -r requirements.txt`
 - Install PyQt6 (`pip install PyQt6`)
 - Change the line which sets the value `DEVICE` to either `cpu` or `cuda:[gpu_num]` where `[gpu_num]`is the number of the GPU you intend to use. 
 - Change the line which sets the value `CHECKPOINT_PATH` to the path of the model file. By default this path is set to the relative path `weights/full_model_skipclick.pth`. You can download the model file for the full SkipClick model [here](https://mediastore.rz.uni-augsburg.de/get/MQjMyFYnZ7/).
 - Optional: You can also choose to change the canvas size by modifying the variables `H_CANVAS_MAX` and `W_CANVAS_MAX`.

Move to the folder that contains the SkipClick repository. The GUI demo can then be run as `python -m novel_demo.gui_demo_qt6`. 

Afterwards, you can use the GUI in the following way: 
 - Press the `l` key to load an image. 
 - Use the left / right mouse button to place a foreground / background click. 
 - Press the `u` key to undo the last click. 
 - Press the `s` key to save the current mask (as a `.png` file). 