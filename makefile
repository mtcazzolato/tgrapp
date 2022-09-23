# goal: run demo for tgrapp

INPUT_DIR=data
SAMPLE_FILE=sample_raw_data.csv

run:
	streamlit run app/tgrapp.py --server.maxUploadSize 5000

prep:
	pip3 install -r requirements.txt
