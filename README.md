### Python environment setup with Conda

```bash
conda create -n gt python=3.9
conda activate gt

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

pip install ogb
pip install pygsp
pip install scipy

conda clean --all
```


### Preprocess
```bash
conda activate gt
python3 preprocess_data.py
```

As for larger dataset preprocessing, consider using preprocess_data_large.py.
The graph coarsening is not used in preprocess_data_large.py due to its instability.

### Training
```bash
sh cora.sh
sh arxiv.sh
```
