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


### Compile
```bash
cd Precompute/ && python setup.py build_ext --inplace && cd ..
```

### Run

```bash
python main.py
```
