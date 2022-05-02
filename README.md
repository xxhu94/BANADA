To run the code, you need to have at least Python 3.7 or later versions.  
## Running BANADA
1.In BANADA/data directory，run`unzip BUPT.zip` and `unzip Sichuan.zip` to unzip the datasets;  
2.Run `python data_process.py` to generate Sichuan and BUPT dataset in DGL;  
3.Run `python main.py` to run BANADA with default settings.  
For other dataset and parameter settings, please refer to the arg parser in train.py. Our model supports both CPU and GPU mode.  

## Repo Structure
The repository is organized as follows:
- `baselines/`:code for all the baselines used in our paper;  
- `data/`: dataset files;  
- `data_process.py`: convert raw node features and adjacency matrix to DGL dataset;
- `main.py`: training and testing BANADA;
- `model.py`: BANADA model implementations;
- `utils.py`: utility functions for EarlyStopping,MixedDropout and MixedLinear.  


## Running baselines
You can find the baselines in `baselines` directory. For example, you can run Player2Vec using:
```bash
python Player2Vec_main.py 
```

  