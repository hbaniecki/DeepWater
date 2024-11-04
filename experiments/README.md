## Setup

```bash
conda create -n deepwater python=3.9

conda activate deepwater

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install ipykernel pandas numpy==1.26.4 captum==0.7.0 torcheval dattri pyDVL[influence]
```