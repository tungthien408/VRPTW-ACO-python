from vrptw_base import VrptwGraph
from multiple_ant_colony_system import MultipleAntColonySystem


if __name__ == '__main__':
    file_path = './dataset/solomon_25/C101.txt'
    ants_num = 10
    beta = 2
    q0 = 0.1
    show_figure = False 
    max_iterations = 50

    graph = VrptwGraph(file_path)
    macs = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0, whether_or_not_to_show_figure=show_figure)
    macs.run_multiple_ant_colony_system()
