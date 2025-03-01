# HubGT

This is the original implementation of *HubGT: Scalable Graph Transformer with Decoupled Hierarchy Labeling*.

Tech report is available at [Appendix.pdf](Appendix.pdf).

## Data Preparation
Datasets are automatically downloaded by the code and stored in the `data` folder.

## Environment setup 

Install by Conda:

```bash
conda create -n gt python=3.12
conda activate gt

conda install pytorch=2.2 -c pytorch -c nvidia
conda install pyg=2.5.3 -c pyg -c conda-forge
conda install scipy optuna tqdm torchmetrics pandas scikit-learn cython=3.0
conda install -c conda-forge ogb
```

## Precomputation

* Environment: CMake 3.16, C++ 14. Compile the Cython code for precomputation:

```bash
cd Precompute/ && python setup.py build_ext --inplace && cd ..
```

* Index is stored in the `cache` folder.

## Run the experiments

* Run training and evaluation:

```bash
python main.py --data [dataset] --batch [batch_size] -ss [total size] -s0 [out label size] -s0g [global label size] -s1 [in label size]
```

For other configurations refer to: `python main.py --help`

* Use script `scripts/tune.sh` to tune hyperparameters, script `scripts/run.sh` to run the code with the best hyperparameters.

* Results are stored in the `log` folder, summary is stored in `log/summary.csv`.
