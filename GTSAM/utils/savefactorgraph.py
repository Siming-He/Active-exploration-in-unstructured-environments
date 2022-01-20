import graphviz
import os

def save_factor_graph(graph, name):
    """
    Save factor graph as got and png files

    Args:
        graph (NonlinearFactorGraph)
        name (string): the name of output files
    """
    path = f"{name}Outputs"
    
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    graph.saveGraph(f"{path}/{name}.dot")
    g = graphviz.Source.from_file(f"{path}/{name}.dot")
    g.format = 'png'
    g.render(directory=f"{name}Outputs").replace('\\', '/')
