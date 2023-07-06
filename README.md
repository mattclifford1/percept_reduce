# set up
```
conda create -n percept python=3.10 -y
conda activate percept
```
get pytorch 2.0.1
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```
```
pip install -r requirements.txt
```
```
pip install -e .
```

or all in one line (make sure your in the repo):
```
conda create -n percept python=3.10 -y && conda activate percept && conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y && pip install -r requirements.txt && pip install -e .
```
