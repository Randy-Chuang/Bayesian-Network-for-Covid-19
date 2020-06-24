# -*- coding: utf-8 -*-
# @Filename : process.py
# @Description: Frequently used function
# @Date : 2020-June
# @Project: Early detection of Covid-19 using BN (AI Term project)
# @AUTHOR : Randy
from pgmpy.models import BayesianModel


def saveGraphToPDF(file_name: str, Edge_list: list, Is_DAG: bool) -> bool:
    """
    A function used to save the graph into PDF file.
    Issue: Haven't found how to fix the position of particular vertex 
            in order to compare different network with identical set of vertices.
    """
    from graphviz import Graph
    from graphviz import Digraph

    if(Is_DAG):
        dot = Digraph(comment="Target Bayesian Network", format="pdf")
    else:
        dot = Graph(comment="Graph", format="pdf")

    # for level in vertices_list:
    #     with dot.subgraph() as s:
    #         s.attr(rank='same')
    #         for vertex in level:
    #             s.node(vertex)
    
    Edge_list.sort()
    for arc in Edge_list:
        # print(arc)
        dot.edge(arc[0], arc[1])
    # print(dot.source)

    try:
        dot.render(file_name, view=True)
    except OSError:
        print("Permission denied: Saving file to: ", file_name)
        return False
    else:
        print("Successfully save Graph into PDF file!")
        return True


def saveModel(model: BayesianModel, file_name: str) -> bool:
    """
    A function used to save the given model to BIF file format.
    """
    from pgmpy.readwrite import BIFReader, BIFWriter
    writer = BIFWriter(model)
    
    try:
        writer.write_bif(filename=file_name)
    except OSError:
        print("Permission denied: Saving file to: ", file_name)
        return False
    else:
        print("Successfully save model to: ", file_name)
        return True
