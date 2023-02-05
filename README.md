
# TgrApp

*TgrApp*: Anomaly Detection and Visualization of Large-Scale Call Graphs  

**Authors:** Mirela T. Cazzolato<sup>1,2</sup>, Saranya Vijayakumar<sup>1</sup>, Xinyi Zheng<sup>1</sup>, Namyong Park<sup>1</sup>, Meng-Chieh Lee<sup>1</sup>, Duen Horng Chau<sup>3</sup>, Pedro Fidalgo<sup>4,5</sup>, Bruno Lages<sup>4</sup>, Agma J. M. Traina<sup>2</sup>, Christos Faloutsos<sup>1</sup>.  

**Affiliations:**  <sup>1</sup> Carnegie Mellon University (CMU), <sup>2</sup> University of São Paulo (USP), <sup>3</sup> Georgia Institute of Technology, <sup>4</sup> Mobileum, <sup>5</sup> ISCTE-IUL  
    
**Conference:** [The 37th AAAI Conference on Artificial Intelligence (AAAI), 2023 @ Washington DC, USA.](https://aaai.org/Conferences/AAAI-23/)  

Please cite the paper as (to appear):  

```
@inproceedings{cazzolato2023tgrapp,
  title={{TgrApp}: Anomaly Detection and Visualization of Large-Scale Call Graphs},
  author={Cazzolato, M.T. and Vijayakumar, S. and Zheng, X. and Park, N. and Lee, M-C. and Chau, D.H. and Fidalgo, P. and Lages, B. and Traina, A.J.M. and Faloutsos, C..},
  booktitle={The 37th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2023},
  note={To appear}
}
```

## Requirements

Check file `requirements.txt`  

To create and use a virtual environment, type:  

    python -m venv tgrapp_venv  
    source tgrapp_venv/bin/activate  
    pip install -r requirements.txt  

## Instructions for M1 / Arm computers

For streamlit app locally on M1:  

    conda create --name tgrapp python=3.8  
    conda install scikit-learn==0.24.2  

Comment out the scikit learn line in the requirements file (requirements.txt)  
And run:  

    pip install -r requirements.txt  

## Running the app

Run the app with the following command on your Terminal:  

make  
or  

streamlit run app/tgrapp.py --server.maxUploadSize 8000  

- Parameter `[--server.maxUploadSize 8000]` is optional, and it is used to increase the size limit of input files.  

## Data Sample

We provide a toy sample dataset in folder *data/*. Check file *sample_raw_data.csv*  

## Acknowledgement

**Matrix cross-associations**

The code for generating matrix cross-associations is originally from [this Github repository]([https://github.com/clifflyon/fully-automatic-cross-associations](https://github.com/clifflyon/fully-automatic-cross-associations)).  

The work was proposed in [this]([https://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd04-cross-assoc.pdf](https://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd04-cross-assoc.pdf)) paper:  

> Deepayan Chakrabarti, S. Papadimitriou, D. Modha, C. Faloutsos.  
>  **Fully automatic cross-associations**. Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data  
> mining. 2004. DOI:10.1145/1014052.1014064.  

**Anomaly detection with gen<sup>2</sup>Out**  

The code for Gen<sup>2</sup> is originally from [this Github repository]([https://github.com/mengchillee/gen2Out](https://github.com/mengchillee/gen2Out)).  
The work was proposed in [this]([https://arxiv.org/pdf/2109.02704.pdf](https://arxiv.org/pdf/2109.02704.pdf)) paper:  

> Lee, MC., Shekhar, S., Faloutsos, C., Hutson, TN., and Iasemidis, L., **gen2Out: Detecting and Ranking Generalized Anomalies**. _IEEE International Conference on Big Data (Big Data)_, 2021.  

## Short demo video

https://user-images.githubusercontent.com/8514761/192572486-d8449a34-e9ea-4cf0-92e9-1a14dd5a4766.mp4


