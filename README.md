# TgrApp  
Anomaly Detection and Visualization of Large-Scale Call Graphs  
  
  
**Authors:** Mirela T. Cazzolato<sup>1,2</sup>, Saranya Vijayakumar<sup>1</sup>, Xinyi Zheng<sup>1</sup>, Namyong Park<sup>1</sup>, Meng-Chieh Lee<sup>1</sup>, Duen Horng Chau<sup>3</sup>, Pedro Fidalgo<sup>4,5</sup>, Bruno Lages<sup>4</sup>, Agma J. M. Traina<sup>2</sup>, Christos Faloutsos<sup>1</sup>.  
  
**Affiliations:** <sup>1</sup> Carnegie Mellon University (CMU), <sup>2</sup> University of São Paulo (USP), <sup>3</sup> Georgia Institute of Technology, <sup>4</sup> Mobileum, <sup>5</sup> ISCTE-IUL  
  
  
## Requirements  
  
Check file `requirements.txt`  
  
To create and use a virtual environment, type:  
  
python -m venv tgrapp_venv  
source tgrapp_venv/bin/activate  
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
> **Fully automatic cross-associations**. Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data  
> mining. 2004. DOI:10.1145/1014052.1014064.  
  
**Anomaly detection with gen<sup>2</sup>Out**  
  
The code for Gen<sup>2</sup>Out is originally from [this Github repository]([https://github.com/mengchillee/gen2Out](https://github.com/mengchillee/gen2Out)).  
The work was proposed in [this]([https://arxiv.org/pdf/2109.02704.pdf](https://arxiv.org/pdf/2109.02704.pdf)) paper:  
  
> Lee, MC., Shekhar, S., Faloutsos, C., Hutson, TN., and Iasemidis, L., **gen2Out: Detecting and Ranking Generalized Anomalies**. _IEEE International Conference on Big Data (Big Data)_, 2021.  
  
----------------------  
  
# TgrApp: Video tutorial  
  
Step-by-step tutorial on how to use TgrApp to generate features, visualize the results, and dive deep into suspicious nodes/phone numbers.
