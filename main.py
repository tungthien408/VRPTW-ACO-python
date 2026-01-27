# import pandas as pd
# import matplotlib.pyplot as plt

from vrptw_base import VrptwGraph
from basic_aco import BasicACO


# file_path = './dataset/homberger_200_customer_instances/R1_2_1.txt'
if __name__ == '__main__':
    
    input_path = './dataset/'
    result_path = './results/param4/v2/rc/rc2/'

    file_path_solomon = ['solomon_25/', 'solomon_50/', 'solomon-100/']
    file_path_homberger = [
        'homberger_200_customer_instances/', 
        'homberger_400_customer_instances/', 
        'homberger_600_customer_instances/', 
        'homberger_800_customer_instances/', 
        'homberger_1000_customer_instances/'
        ]
    input_filename_solomon = 'rc201.txt'
    input_filename_homberger = [
        'RC2_2_1.txt', 
        'RC2_4_1.txt', 
        'RC2_6_1.txt', 
        'RC2_8_1.txt', 
        'RC2_10_1.txt'
        ]
    output_filename_solomon = ['aco_results_25.csv', 'aco_results_50.csv', 'aco_results_100.csv']
    output_filename_homberger = [
        'aco_results_200.csv', 
        'aco_results_400.csv', 
        'aco_results_600.csv', 
        'aco_results_800.csv', 
        'aco_results_1000.csv'
        ]

    ants_num = 20
    max_iter = 200
    alpha = 1
    beta = 2
    rho = 0.1
    q0 = 0.1
    show_figure = False 

    # solomon
    for filepath, filename in zip(file_path_solomon, output_filename_solomon):
        file_path = input_path + filepath + input_filename_solomon
        graph = VrptwGraph(file_path, rho=rho)
        basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0, result_path=result_path, filename=filename,
                        whether_or_not_to_show_figure=show_figure)
        basic_aco.run_basic_aco()
    # homberger
    for filepath, input_filename, output_filename in zip(file_path_homberger, input_filename_homberger, output_filename_homberger):
        file_path = input_path + filepath + input_filename
        graph = VrptwGraph(file_path, rho=rho)
        basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0, result_path=result_path, filename=output_filename,
                        whether_or_not_to_show_figure=show_figure)
        basic_aco.run_basic_aco()


# basic_aco.run_basic_aco()
