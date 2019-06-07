# semantic-video-segmentation
Enhancing temporal consistency in semantic video segmentation, using recurrent neural networks.

## Requirements
- Python 3.7 (not tested with earlier versions)
- See requirements.txt  (TODO remove unnecessary ones)

## Usage

First, download (one of) the datasets:
- CamVid:
[download link for images](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), 
[download link for videos](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).
- Cityscapes: 
[download link](https://www.cityscapes-dataset.com/).

Then, update the dataset configuration file `data/dataset/config.py` to match your directory structure.

### Training a network:
For more info about arguments:
```bash
python train.py --help
```
Example:
```bash
python train.py -c config/enetlstm-cityscapes/ksize3-concat_cuda.json --cuda
```
The training results will appear under saved/<config_name>/. 

To use a specific GPU, put `CUDA_VISIBLE_DEVICES=<device index>` in front of the command. 

### Evaluating a network: 
For more info about arguments: 
```bash
python evaluate.py --help
```
Example: 
```bash
python evaluate.py config/visualize/cityscapes-enet-enetlstm.json -v -t 0.1 --cuda
```
The `-v` option visualizes the network(s) on the videos or images in the dataset, as specified in the config file.
To quit the visualization environment, press `q`. Example: 

![alt text](example.png "Example image")

## References

- Code structure is based on [Pytorch template from Victor Huang](https://github.com/victoresque/pytorch-template).
- ENet is heavily based on the [ENet implmenetation of David Silva](http://www.dropwizard.io/1.0.2/docs/).
- ConvLSTM is heavily based on the [ConvLSTM implementation of Andrea Palazzi](https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py).
- ConvGRU is heavily based on the [ConvGRU implementation of Jacob Kimmel](https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py).
