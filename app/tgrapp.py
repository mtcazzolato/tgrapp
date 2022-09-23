# Author: Mirela Cazzolato
# Date: September 2022
# Goal: Generate and visualize features from t-graphs for anomaly detection
# =======================================================================

import streamlit as st

# WCW-App windows
from window_feature_extraction import launch_w_feature_extraction
from window_hexbin import launch_w_hexbin
from window_lasso import launch_w_lasso
from window_scatter_matrix import launch_w_scatter_matrix
from window_deep_dive import launch_deep_dive
from window_negative_list import launch_w_negative_list
from window_attention_routing import launch_w_attention_routing

# Change page width and configure on load
st.set_page_config(layout="wide",
                   page_icon="📈",
                   page_title="TgrApp",
                   initial_sidebar_state="auto"
)

# Available taks
w_tasks = ["Feature extraction",
           "Hexbin scatter plot",
           "Lasso selection and deep dive",
           "Interactive scatter matrix",
           "Deep dive on calls over time",
           "Attention routing",
           "Manage negative-list"
]

with st.sidebar:
    st.write(
        """
        # 📈 TgrAPP
        Anomaly detection and visualization of large-scale call graphs.
        """
    )

    # List tasks on the sidebar
    selectbox_window_task = st.sidebar.selectbox(
        "Select one of the following tasks:",
        (w_tasks)
    )
    
# Open file of the selected task from sidebar
if (selectbox_window_task == str(w_tasks[0])):
    launch_w_feature_extraction()
elif (selectbox_window_task == w_tasks[1]):
    launch_w_hexbin()
elif (selectbox_window_task == w_tasks[2]):
    launch_w_lasso()
elif (selectbox_window_task == w_tasks[3]):
    launch_w_scatter_matrix()
elif (selectbox_window_task == w_tasks[4]):
    launch_deep_dive()
elif (selectbox_window_task == w_tasks[5]):
    launch_w_attention_routing()
elif (selectbox_window_task == w_tasks[6]):
    launch_w_negative_list()
else:
    st.write("# Sorry, tool not available yet")
    
# run from Terminal:
# streamlit run repo/wcw_app.py --server.maxUploadSize 10000
