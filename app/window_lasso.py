# Author: Mirela Cazzolato
# Date: July 2022
# Goal: Window to select raw data and generate t-graph features
# =======================================================================

import streamlit as st

import os.path

import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as py
from ipywidgets import interactive, HBox, VBox
from streamlit_plotly_events import plotly_events

from collections import Counter
from cross_associations.matrix import Matrix
from cross_associations.plots import plot_shaded, plot_spy
import cross_associations.cluster_search as cs

sns.set()

file_negative_list="data/negative-list.csv"
NODE_ID="node_ID"
SOURCE="source"
DESTINATION="destination"
MEASURE="measure"
flag_graph_constructed=False
plotly_width="100%"
plotly_height=800


def read_files(file_features, file_graph):
    """
    Read files with features and raw data
    
    Parameters
    ----------
    file_features: str
        path of the input file with t-graph features
    file_graph: str
        path of the input file with raw data
    """


    global df, df_graph

    # File with input features
    df = pd.read_csv(file_features)

    # Select nodes with more than 5 calls
    # df = df[(df['in_call_count'] + df['out_call_count']) >= 5].reset_index(drop=True)
    
    # File with raw data (source, destination, measure, timestamp) to generate the graph
    df_graph = pd.read_csv(file_graph)

    filter_negative_list()

    df.fillna(0, inplace=True)
    flag_graph_constructed=False
    
def update_sidebar():
    """
    Add options to the sidebar
    """

    global df, ego_radius, max_nodes_association_matrix
    with st.sidebar:
        ego_radius = st.number_input(
            "EgoNet radius parameter",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            format="%d",
            help="""Radius of the egonet. It
                  may take a while to run. Use with caution."""
        )

        max_nodes_association_matrix = st.number_input(
            "Max #nodes for matrix association",
            min_value=1,
            # max_value=5,
            value=500,
            step=20,
            format="%d",
            help="""Maximum number of nodes allowed to generate the
                  matrix association plot. With high #nodes, it
                  may take a while to run. Use with caution."""
        )

    # TODO
    # # Add option to sort data
    # add_sort_field()


def add_sort_field():
    """
    Add field to sort values in the sidebar
    """

    global df
    with st.sidebar:
        selectbox_sorting_column = st.sidebar.selectbox(
            "Sort data by (descending order):",
            options=df.columns,
            index=0
        )
        
        if st.button('Sort features'):
            print('sort by', selectbox_sorting_column)
            df = df.sort_values(by=selectbox_sorting_column, ascending=False)


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


def populate_selectbox_graph():
    """
    Populate select box with available columns for the graph
    """

    global df_graph, SOURCE, DESTINATION, MEASURE
    mcol1_graph, mcol2_graph, mcol3_graph = st.columns(3)

    with mcol1_graph:
        SOURCE = st.selectbox(
                        "Select SOURCE column",
                        options=df_graph.columns,
                        index=0)
    with mcol2_graph:
        DESTINATION = st.selectbox(
                        "Select DESTINATION column",
                        options=df_graph.columns,
                        index=1)
    with mcol3_graph:
        MEASURE = st.selectbox(
                        "Select MEASURE column",
                        options=df_graph.columns,
                        index=2)


def construct_graph():
    """
    Construct graph for the deep dive with selected attributes
    """

    global G, SOURCE, DESTINATION, MEASURE, flag_graph_constructed
    G = nx.from_pandas_edgelist(df_graph, source=SOURCE,
                                            target=DESTINATION,
                                            edge_attr=MEASURE,
                                            create_using=nx.DiGraph())

    flag_graph_constructed=True


def filter_negative_list():
    """
    Remove nodes included in the negative list
    """
    global df, df_graph

    if os.path.isfile(file_negative_list):
        # Remove nodes in the negative-list
        df_negative_list = pd.read_csv(file_negative_list)

        df = df[~df[NODE_ID].isin(list(df_negative_list[NODE_ID].values))].reset_index(drop=True)

        df_graph = df_graph[~df_graph[SOURCE].isin(list(df_negative_list[NODE_ID].values))]
        df_graph = df_graph[~df_graph[DESTINATION].isin(df_negative_list[NODE_ID].values)].reset_index(drop=True)


def plot_scatter():
    """
    Plot interactive scatter plot with the selected pair of features
    """
    
    global df, feature1, feature2
    f = go.FigureWidget([go.Scatter(x = np.log10(df[feature1]+1),
                                    y = np.log10(df[feature2]+1),
                                    mode = 'markers')])
    f.layout.xaxis.title = feature1.replace('_', ' ') + ' -- log10(x+1)'
    f.layout.yaxis.title = feature2.replace('_', ' ') + ' -- log10(x+1)'
    # f.update_layout(width=plotly_width,height=plotly_height)

    scatter = f.data[0]
    scatter.marker.opacity = 0.5

    return f


def plot_adj_matrix(G, markersize=2, compute_associations=True):
    """
    Plot adjacency matrix of the given graph

    Parameters
    ----------
    G: nx.Graph
        graph to show
    markersize: int
        size of the marker to show the correspondences in the plot
    compute_associations: boolean
        informs if the matrix associations should be generated
    """

    fig, ax = plt.subplots()
    plt.spy(nx.adjacency_matrix(G), markersize=markersize)
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel('source')
    plt.xlabel('destination')
    # plt.tight_layout()

    if compute_associations: # Compute matrix associations
        fig_cross_associations = plot_cross_associations(nx.adjacency_matrix(G))
    else:
        return fig  # Skip matrix associations and return only the original matrix

    return fig, fig_cross_associations


def plot_cross_associations(sparse_matrix, markersize=2):
    """
    Find cross associations in the given adjacency matrix

    Parameters
    ----------
    sparse_matrix: matrix
        adjacency matrix to find cross associations
    markersize: int
        size of the marker to show the correspondences in the plot
    """

    # Run algorithm to find cross associations
    matrix = Matrix(sparse_matrix)
    cluster_search = cs.ClusterSearch(matrix)
    cluster_search.run()
    col_counter = Counter(cluster_search._matrix.col_clusters)
    row_counter = Counter(cluster_search._matrix.row_clusters)
    height, width = np.shape(cluster_search._matrix.matrix)

    # Plot results
    fig_cross_associations = plt.figure()
    ax1 = fig_cross_associations.add_subplot(111)
    ax1.set_xlim(right=width)
    ax1.set_ylim(top=height)
    ax1.set_ylabel('source')
    ax1.set_xlabel('destination')
    col_offset = 0
    for col, col_len in col_counter.most_common():
        row_offset = 0
        for row, row_len in row_counter.most_common():

            ax1.add_patch(
                patches.Rectangle(
                    (col_offset * 1.0, row_offset * 1.0),
                    (col_offset + col_len) * 1.0,
                    (row_offset + row_len) * 1.0,
                    facecolor=None,
                    color=None,
                    edgecolor="#0000FF",
                    linewidth=0.5,
                    fill=False
                )
            )
            row_offset += row_counter[row]
        col_offset += col_counter[col]

    ax1.spy(cluster_search._matrix.transformed_matrix, markersize=markersize, color='black')
    ax1.set_aspect('equal', adjustable='box')

    return fig_cross_associations


def get_egonet(G, suspicious_nodes, radius=1):
    """
    Compose a graph with the egonets of a given set of suspicious nodes.
    Return the subgraph of the composed egonets and the index of
    suspicious nodes inside the subgraph

    Parameters
    ----------
    G: nx.Graph
        graph with all nodes to extract the EgoNets from
    suspicious_nodes: list
        list of suspicious nodes
    radius: int
        step of the EgoNet
    """
    
    final_G = nx.empty_graph(create_using=nx.DiGraph())

    for ego_node in suspicious_nodes:
        # create ego network
        hub_ego = nx.ego_graph(G, ego_node, radius=radius, distance='weight', undirected=True)
        final_G = nx.compose(final_G, hub_ego)

    idx_suspicious_nodes = []
    for node in suspicious_nodes:
        idx_suspicious_nodes.append(list(np.where(pd.DataFrame(data=final_G.nodes()) == node)[0])[0])
    
    return final_G, idx_suspicious_nodes


def plot_parallel_coordinates(df_features, columns):
    """
    Plot parallel ploting of features and the given columns


    Parameters
    ----------
    df_features: DataFrame
        features to plot
    columns: list
        columns to use as coordinates
    """
    # TODO: adjust size and details of the plot
    
    fig = px.parallel_coordinates(df_features,
                                  dimensions=columns,
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2)
    fig.update_layout(#width=900,
                      height=550)
    fig.update_layout(
        font_size=22
    )

    return fig


def plot_graph_spring_layout(G):
    """
    Plot given graph using the spring layout

    Parameters
    ----------
    G: Graph
        Graph object from networkx
    """

    fig = plt.figure()

    nx.draw(G,
            with_labels=False,
            node_size=50,
            node_color="skyblue",
            pos=nx.spring_layout(G))

    # plt.title("Generated EgoNet")
    
    return fig


def plot_interactive_egonet(G, suspicious_nodes=[]):
    """
    Plot given graph using the spring layout

    Parameters
    ----------
    G: Graph
        Graph object from networkx
    suspicous_nodes: list
        List of node ids to be painted in red
    """
    
    edge_x = []
    edge_y = []
    pos = nx.layout.spring_layout(G)
    
    
    for edge in G.edges():
        x0, y0 = pos.get(edge[0])
        x1, y1 = pos.get(edge[1])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos.get(node)
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y,
        mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=15,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    ),
            line_width=2))

    node_adjacencies = []
    node_text = []

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1])) # number of adjacencies to use as color
        node_text.append('Node ID: ' + str(list(G.nodes())[node]) + ', # of connections: ' + str(len(adjacencies[1])))
    
    for s in suspicious_nodes:
        node_adjacencies[s] = 'red'

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        #title='Deep dive on suspicious nodes',
                        titlefont_size=16, showlegend=False, hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            showarrow=True,
                            xref="paper",
                            yref="paper",
                            x=0.005, y=-0.002 ) ],
                    # width=1200,
                    height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def launch_w_lasso():
    """
    Launch window to visualize features interactively,
    and do deep dive with selected nodes
    """

    st.write(
        """
        # Lasso selection
        ### Visualize features and select nodes for a deep dive
        """
    )
    
    with st.expander(label="Input data", expanded=True):
        selected_points = []
        df_result = pd.DataFrame()
        col1_file_selection, col2_file_selection = st.columns(2)

        with col1_file_selection:
            file_features = st.file_uploader(label="Select a file with features",
                                    type=['txt', 'csv'])

            use_example_features = st.checkbox("Use example file with features",
                                    False,
                                    help="Use in-built example file with features to demo the app")

        with col2_file_selection:
            file_graph = st.file_uploader(label="Select a file with raw data",
                                    type=['txt', 'csv'])
                                    
            use_example_graph = st.checkbox("Use example file with raw data",
                                    False,
                                    help="Use in-built example file of raw data to demo the app")
        

        if use_example_features and not file_features:
            file_features = "data/allFeatures_nodeVectors.csv"

        if use_example_graph and not file_graph:
            file_graph = "data/sample_raw_data.csv"
    
        if file_features and file_graph:
            read_files(file_features, file_graph)

            populate_selectbox_graph()

            if st.button('Construct graph'):
                construct_graph()
    
    if flag_graph_constructed:
        update_sidebar()

        with st.expander(label="Lasso selection", expanded=True):
            populate_selectbox()

            if not feature1:
                st.error("Please select feature 1")
            elif not feature2:
                st.error("Please select feature 2")
            else:
                # st.write("### Select nodes of interest")

                fig = plot_scatter()
                # Add interactive scatter plot
                selected_points = plotly_events(fig, select_event=True,
                                                override_height=plotly_height,
                                                override_width=plotly_width,)
            
        with st.expander(label="Deep dive on selected nodes", expanded=True):
            if len(selected_points) > 0:
                df_selected = pd.DataFrame(selected_points)
                
                st.write("Selected nodes:", len(selected_points))
                st.write("### Features of select nodes")
                
                st.dataframe(df.loc[df_selected["pointNumber"].values])
                
                st.write("### Adjacency matrix of the generated EgoNet")

                final_G, idx_suspicious_nodes = get_egonet(G,
                                        suspicious_nodes=df.loc[df_selected["pointNumber"].values][NODE_ID],
                                        radius=ego_radius, #2-step way egonet
                                        )
                
                st.write("EgoNet size:", len(final_G.nodes))
                
                if len(final_G.nodes) < max_nodes_association_matrix:
                    fig_adj_matrix, fig_cross_associations = plot_adj_matrix(G=final_G, compute_associations=True)
                else:
                    fig_adj_matrix = plot_adj_matrix(G=final_G, compute_associations=False)

                col1, col2 = st.columns([1, 1])
                col1.pyplot(fig_adj_matrix)
                
                if len(final_G.nodes) < max_nodes_association_matrix:
                    col2.pyplot(fig_cross_associations)

                st.write("Nodes in the EgoNet:")
                df_result = pd.DataFrame(data=final_G.nodes, columns=[NODE_ID]).set_index(NODE_ID).join(df.set_index(NODE_ID)).reset_index()
                df_result.columns=df.columns
                st.write(df_result.fillna(0))

        if len(df_result) > 0:
            with st.expander(label="EgoNet visualization (selected nodes in red)", expanded=True):
                fig_plotly_graph_spring_layout = plot_interactive_egonet(final_G,
                                                                         suspicious_nodes=idx_suspicious_nodes)
                st.plotly_chart(fig_plotly_graph_spring_layout,
                                use_container_width=True)

        if len(df_result) > 0:
            with st.expander(label="Parallel coordinates", expanded=True):
                fig_parallel_coords = plot_parallel_coordinates(df_features=df_result, columns=df.columns[1:8])
                st.plotly_chart(fig_parallel_coords,
                                use_container_width=True)

                # selected_points_pc = plotly_events(fig_parallel_coords, select_event=True)
                # st.write(selected_points_pc)
