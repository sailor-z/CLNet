# CLNet
(ICCV 2021) PyTorch implementation of Paper "Progressive Correspondence Pruning by Consensus Learning"
 * [[project page](https://sailor-z.github.io/projects/CLNet.html)]
 * [[paper](https://arxiv.org/abs/2101.00591#)]

# Citing CLNet
If you find the CLNet code useful, please consider citing:

```bibtex
@inproceedings{zhao2021progressive,
  title={Progressive Correspondence Pruning by Consensus Learning},
  author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision.},
  year={2021}
}
```

# Setup
Please start by installing the required libraries:

    pip install -r requirements.txt

# Data Processing
The code of this part is partially borrowed from [[OANet](https://github.com/zjhthu/OANet)] [[CNe](https://github.com/vcg-uvic/learned-correspondence-release)]. Please follow their instructions to download the training and testing data.

    bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8 ## YFCC100M
    tar -xvf raw_data_yfcc.tar.gz

    bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2 ## SUN3D
    tar -xvf raw_sun3d_test.tar.gz
    bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
    tar -xvf raw_sun3d_train.tar.gz

After downloading the datasets, the initial matches can be generated by:

    cd dump_match
    bash yfcc.sh
    bash sun3d.sh

The initial matches are generated over SIFT by default. The ones based on ORB and SuperPoint are also available by changing the settings of `--suffix` and `--desc_name`.

## Pretrained Model

We provide a pretrained model on YFCC100M. The results in our paper can be reproduced by running the test script:

    python ./test.py --use_ransac True --data_te ./data_dump/yfcc-sift-2000-test.hdf5 --output_dir ./logs/CLNet_yfcc_sift --model_path ./pretrained_models/clnet_yfcc_sift.pth

## Train model on YFCC100M
Please run the training script to train our model on YFCC100M after the data processing is done.

    python ./train.py --data_tr ./data_dump/yfcc-sift-2000-train.hdf5
    --data_te ./data_dump/yfcc-sift-2000-test.hdf5
