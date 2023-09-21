# hr_GPN_SNN
This is the code for article "Gated Parametric Neuron for Spiking Neural Networks"
## Code structure
* **core**
  * **audio_dataset.py**: dataset generation
  * **losses.py**: loss functions
  * **methods.py**: a dropout module
  * **surrogate.py**: surrogate function
  * **tools.py**: other tools
* **src**
    * **model.py**: SNN models
    * **SHD.py**: run file for SHD
    * **SSC.py**: run file for SSC
## Experiment
1. Dataset generation
   ```
   python audio_dataset.py  -T=20 -data_name='SHD'
   python audio_dataset.py  -T=60 -data_name='SSC'
   ```
2. Run file
   ```
   CUDA_VISIBLE_DEVICES=1 python SHD.py -T=20 -neuron_func=GPN -loss_func=mean -repeat=3 -path_name=SHD_GPN
   CUDA_VISIBLE_DEVICES=1 python SSC.py -T=60 -neuron_func=GPN -loss_func=mean -repeat=3 -path_name=SSC_GPN
   ```
## Note
Our code is based on [SpikingJelly](https://github.com/fangwei123456/spikingjelly).
