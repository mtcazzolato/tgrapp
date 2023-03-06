from cProfile import run
from lib2to3.pgen2.pgen import DFAState
from tkinter import NoDefaultRoot
from tkinter.tix import Tree
import streamlit as st

from gen2Out.gen2out import gen2Out
from gen2Out.utils import sythetic_group_anomaly, plot_results
from gen2Out.iforest import IsolationForest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from ipywidgets import interactive, HBox, VBox
from streamlit_plotly_events import plotly_events

from sklearn import manifold

sns.set()

NODE_ID="node_ID"
LABEL=""
NODE_ID_LABEL=""
LABEL_TRUE_VALUE=""

num_labels=2
labels=['False', 'True']
flag_labels_sorted=False
plotly_width="100%"
plotly_height=800


def read_file(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with features
    """

    global df, nrows
    df = pd.read_csv(file)
    df.fillna(0, inplace=True)


def read_file_label(file):
    """
    Read columns from input file
    
    Parameters
    ----------
    file: str
        path of the input file with labels
    """

    global df_label, nrows, flag_labels_sorted
    df_label = pd.read_csv(file)
    flag_labels_sorted = False


def sort_labels(label_column_node_id, label_column_name, label_true_value):
    """
    Sort input labels by NODE_ID, according to the order
    given by the nodes
    
    Parameters
    ----------
    label_column_node_id: str
        column from the lables' file with the NODE_ID
    label_column_name: str
        column from the lables' file with label values
    label_true_value: str
        string with the value corresponding to TRUE for anomaly
    """

    global df, df_label, NODE_ID_LABEL, LABEL, LABEL_TRUE_VALUE, flag_labels_sorted

    NODE_ID_LABEL = label_column_node_id
    LABEL = label_column_name
    LABEL_TRUE_VALUE = label_true_value
    
    # Get unique node values for True and False
    true_nodes = df_label[df_label[LABEL].astype(str) == str(LABEL_TRUE_VALUE)][NODE_ID_LABEL].unique()
    df_label_unique = pd.DataFrame(data=true_nodes, columns=[NODE_ID_LABEL])
    
    LABEL_TRUE_VALUE = 'True'
    df_label_unique[LABEL] = len(df_label_unique) * [LABEL_TRUE_VALUE]

    # Join feature df with unique labeled dataframe
    df_label = df.set_index(NODE_ID).join(df_label_unique.set_index(NODE_ID_LABEL), how='left')[[LABEL]]
    df_label = df_label.reset_index()
    
    # Replace NaN with False
    df_label[LABEL].fillna('False', inplace=True)

    NODE_ID_LABEL = NODE_ID

    flag_labels_sorted = True


def update_sidebar():
    """
    Add options to the sidebar
    """
    
    global df, nrows

    # TODO


def run_tsne(columns, n_components=2, perplexity=100):
    """
    Run t-SNE over the input data

    Parameters
    ----------
    columns: array of strings
        column names to be used
    n_components: int
        number of components
    perplexity: str
        related to the number of nearest neighbors
        that is used in other manifold learning algorithms.
        See Sklearn documentation for details.
    """
    
    global df, df_label
    tsne = manifold.TSNE(
                n_components=n_components,
                init="random",
                random_state=42,
                perplexity=perplexity,
                n_iter=300,
            )
    df_tsne = tsne.fit_transform(df[columns])
    df_tsne = pd.DataFrame(data=df_tsne, columns=['x', 'y'])
    
    return df_tsne


def plot_tsne_interactive(columns, n_components=2, perplexity=100):
    """
    Plot the interactive version of t-SNE plot

    Parameters
    ----------
    columns: array of strings
        column names to be used
    n_components: int
        number of components
    perplexity: str
        related to the number of nearest neighbors
        that is used in other manifold learning algorithms.
        See Sklearn documentation for details.
    """

    global df, df_label
    # df_tsne = run_tsne(columns, n_components, perplexity)
    df_tsne = run_tsne(df.columns[1:], n_components, perplexity)

    # fig = go.FigureWidget([
    #                 go.Scatter(
    #                     x=df_tsne['x'],
    #                     y=df_tsne['y'],
    #                     mode='markers',
    #                     marker=dict(
    #                         color=[int(x==True) for x in df_label[LABEL]],
    #                     ),
    #                     # hover_data=[df[NODE_ID]],
                        
    #                     #  labels={
    #                     #     "x" : "1st Component",
    #                     #     "y" : "2nd Component",
    #                     #     "color": "Fraud",
    #                     #     # "size": "50",
    #                     #  },
                        
    #     )]
    # )
    
    print(df_tsne, df_label)

    fig = go.Figure(
                px.scatter(df_tsne,
                     x='x',
                     y='y',
                     color=df_label[LABEL],
                    #  hover_data=[df_label[NODE_ID]],
                     labels={
                        "x" : '1st Component',
                        "y" : '2nd Component',
                        "color": "Fraud",
                        # "hover_data": "Node ID"
                        # "title": "Scores",
                        # "size": "50",
                     },
                )
    )

    # fig.update_traces(
    #             marker=dict(showscale=False,
    #                             line_width=0.8,
    #                             size=10,
    #                             opacity=0.5),
    #             unselected_marker=dict(opacity=0.1, size=5),
    #                     selected_marker=dict(size=15, opacity=0.9),
    #                     selector=dict(type='splom'),
    #                     diagonal_visible=False
    # )

    fig.update_layout(#title="tSNE",
                      # font_size=20,
                      #   width=1100,
                      #   height=800
                      dragmode='select',
                      hovermode='closest',
    )

    scatter = fig.data[0]
    # scatter.marker.opacity = 0.5

    return fig


def populate_selectbox():
    """
    Populate select box with available features to visualize
    """

    global df, feature1, feature2
    mcol1_features, mcol2_features = st.columns(2)

    with mcol1_features:
        feature1 = st.selectbox("Select first feature",
                                options=df.columns[1:],
                                index=0)
    with mcol2_features:
        feature2 = st.selectbox("Select second feature",
                                options=df.columns[1:],
                                index=1)

def run_gen2out(features):
    """
    Run gen2Out to identify anomalies in the input data

    Parameters
    ----------
    features: list of strings
        List of features (columns) to be used when generating t-SNE
    """

    global df
    model = gen2Out(lower_bound=9,
                    upper_bound=12,
                    max_depth=7,
                    rotate=True,
                    contamination='auto',
                    random_state=0)

    # pscores = model.point_anomaly_scores(X=df[features].values)
    # gscores = model.group_anomaly_scores(X=df[features].values)

    pscores = model.point_anomaly_scores(X=df[df.columns[1:]].values)
    gscores = model.group_anomaly_scores(X=df[df.columns[1:]].values)

    return(plot_results(X=df[features].values, model=model))

def run_iforest(features):
    """
    Run Isolation Forest

    Parameters
    ----------
    features: list of strings
        List of features (columns) to be used when running iForest
    """

    global df
    if_scores = IsolationForest(random_state=42,
                          max_samples=len(df),
                          contamination='auto',
                        #   rotate=True).fit_predict(df[features]) #,max_depth=7
                          rotate=True).fit_predict(df[df.columns[1:]]) #,max_depth=7

    return if_scores

def plot_iforest_scores_interactive(iforest_scores, feature1, feature2):
    """
    Plot Isolation Forest's scores

    Parameters
    ----------
    iforest_scores: list of double values
        List of scores generated by iForest
    feature1:
        First feature to plot (x axis)
    feature2:
        Second feature to plot (y axis)
    """

    global df
    
    fig = go.Figure(px.scatter(df[[feature1, feature2]],
                     x=feature1,
                     y=feature2,
                     color=iforest_scores.astype(str),
                     hover_data=[df[NODE_ID]],
                     labels={
                        "x" : feature1.replace('_', ' '),
                        "y" : feature2.replace('_', ' '),
                        "color": "iForest scores",
                        # "title": "Scores",
                        # "size": "50",
                     },
           )
    )

    fig.update_traces(marker=dict(showscale=False,
                                line_width=0.8,
                                size=10,
                                opacity=0.5),
                        unselected_marker=dict(opacity=0.1, size=5),
                        selected_marker=dict(size=15, opacity=0.9),
                        selector=dict(type='splom'),
                        # selector=dict(mode='markers'),
                        # diagonal_visible=False
    )

    fig.update_layout(
                      dragmode='select',
                      hovermode='closest',
                    )

    fig.layout.xaxis.title = feature1.replace('_', ' ')
    fig.layout.yaxis.title = feature2.replace('_', ' ')

    return fig


def launch_w_attention_routing():
    """
    Plot Isolation Forest's scores

    Parameters
    ----------
    iforest_scores: list of double values
        List of scores generated by iForest
    feature1:
        First feature to plot (x axis)
    feature2:
        Second feature to plot (y axis)
    """

    st.write(
        """
        # Attention routing
        ### Apply tSNE, LookOut and Gen2Out
        """
    )

    selected_points_tsne = []
    
    # Show options to select input files with feature and labels
    col1_data_selection, col2_data_selection = st.columns([1, 1])

    with col1_data_selection:
        file = st.file_uploader(label="Select a file with features*",
                                help="File with extracted features (mandatory)",
                                type=['txt', 'csv'])

        use_example_file = st.checkbox(
            "Use example file", False, help="Use in-built example file to demo the app"
        )

    with col2_data_selection:
        file_labels = st.file_uploader(label="Select a file with labels",
                                       help="File with node labels (optional)",
                                       type=['txt', 'csv'])

        use_example_file_labels = st.checkbox(
            "Use example file", False, help="Use in-built example file with labels to demo the app"
        )

    if use_example_file and not file:
        file = "data/allFeatures_nodeVectors.csv"

    if use_example_file_labels and not file_labels:
        file_labels = "data/sample_raw_data.csv"


    if file:
        read_file(file)
        update_sidebar()


    # Show options for label if the user selects a file
    if file_labels:
        with st.expander(label="tSNE: Setup label options", expanded=True):
            read_file_label(file_labels)

            mcol1_label_options, mcol2_label_options, mcol3_label_options = st.columns([1, 1, 1])

            with mcol1_label_options:
                label_column_node_id = st.selectbox("Select column with NODE ID",
                                        options=df_label.columns,
                                        index=0)
                
            with mcol2_label_options:
                label_column_name = st.selectbox("Select column with LABEL",
                        options=df_label.columns,
                        index=1)
        
            with mcol3_label_options:
                label_true_value = st.selectbox("Select TRUE LABEL VALUE",
                                        options=df_label[label_column_name].unique(),
                                        index=0,
                                        help="Only rows with selected value will be considered fraud (True).  "
                                              + "The remaining rows will be considered as not fraud (False)")
            
                st.write("True value for fraud:", label_true_value)
                    
            if st.button('Set labels and plot tSNE'):
                #   If this is not the first time setting the label columns, reload the file
                if flag_labels_sorted:
                    read_file_label(file_labels)
                
                sort_labels(label_column_node_id, label_column_name, label_true_value)

                with st.spinner("Running tSNE, please wait... (this may take a while)"):
                    # with st.expander(label="tSNE", expanded=True):
                    fig_tsne_interactive = plot_tsne_interactive(columns=df.columns[1:10],
                                                                n_components=2,
                                                                perplexity=50)
                                                                
                # st.plotly_chart(fig_tsne_interactive, use_container_width=True)
                selected_points_tsne = plotly_events(fig_tsne_interactive, select_event=True,
                                                        override_height=plotly_height,
                                                        override_width=plotly_width,)

                if (len(selected_points_tsne) > 0):
                    df_selected_tsne = pd.DataFrame(selected_points_tsne)
                    st.write(df.loc[df_selected_tsne["pointNumber"].values])

    
    if file:
        with st.expander(label="Gen2Out and iForest", expanded=True):
            populate_selectbox()

            if not feature1:
                st.error("Please select feature 1")
            elif not feature2:
                st.error("Please select feature 2")
            else:
                if st.button('Run Gen2Out and iForest'):
                    with st.spinner("Running Gen2Out, please wait... (this may take a while)"):
                        
                        fig_step0_heatmap, fig_step1_xray, fig_step2_apex_extraction, fig_step3_outlier_grouping, fig_step4_anomaly_isocurves, fig_step5_scoring = run_gen2out([feature1, feature2])
                        
                    st.write("Done.")

                    c0_g20, c1_g2o, c2_g2o = st.columns([1, 1, 1])
                    
                    with c0_g20:
                        st.pyplot(fig_step0_heatmap)
                    with c1_g2o:
                        st.pyplot(fig_step1_xray)
                    with c2_g2o:
                        st.pyplot(fig_step2_apex_extraction)
                    with c0_g20:
                        st.pyplot(fig_step3_outlier_grouping)
                    with c1_g2o:
                        st.pyplot(fig_step4_anomaly_isocurves)
                    with c2_g2o:
                        st.pyplot(fig_step5_scoring)

                    with st.spinner('Running iForest, please wait... (this may take a while)'):
                        iforest_scores = run_iforest([feature1, feature2])
    
                    fig_iforest = plot_iforest_scores_interactive(iforest_scores, feature1, feature2)
                    st.plotly_chart(fig_iforest, use_container_width=True)
