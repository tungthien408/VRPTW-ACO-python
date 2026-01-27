import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time


class BasicACO:
    def __init__(self, graph: VrptwGraph, ants_num=10, max_iter=200, alpha=1, beta=2, q0=0.1, 
                 result_path='./results/', filename='aco_results.csv',
                 whether_or_not_to_show_figure=True):
        super()
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.max_load = graph.vehicle_capacity
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.result_path = result_path
        self.filename = filename
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

        self.history = {
            'iteration': [],
            'best_cost': [],
            'best_vehicle_num': [],
            'time_elapsed': [],
            'avg_cost': [],
            'current_iter_best': [],
            'best_path': []  # Thêm lưu lộ trình tốt nhất tại mỗi iteration
        }

    def run_basic_aco(self):
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
        start_time_total = time.time()

        start_iteration = 0
        for iter in range(self.max_iter):

            ants = list(Ant(self.graph) for _ in range(self.ants_num))
            for k in range(self.ants_num):

                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0

                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)

                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)

            paths_distance = np.array([ant.total_travel_distance for ant in ants])

            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1
                start_iteration = iter

                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                print('\n')
                print('[iteration %d]: find a improved path, its distance is %f' % (iter, self.best_path_distance))
                print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))

            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

            self.history['iteration'].append(iter)
            self.history['best_cost'].append(self.best_path_distance if self.best_path_distance else float('inf'))
            self.history['best_vehicle_num'].append(self.best_vehicle_num if self.best_vehicle_num else 0)
            self.history['time_elapsed'].append(time.time() - start_time_total)
            self.history['avg_cost'].append(np.mean(paths_distance))
            self.history['current_iter_best'].append(paths_distance[best_index])
            self.history['best_path'].append(str(self.best_path) if self.best_path else "")

            given_iteration = 100
            if iter - start_iteration > given_iteration:
                print('\n')
                print('iteration exit: can not find better solution in %d iteration' % given_iteration)
                break

        print('\n')
        print('final best path distance is %f, number of vehicle is %d' % (self.best_path_distance, self.best_vehicle_num))
        print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))
        
        import pandas as pd
        import json
        
        # Lưu lịch sử các iteration vào CSV
        df = pd.DataFrame(self.history)
        output = self.result_path + self.filename
        df.to_csv(output, index=False)
        print(f'Results saved to {self.filename}')
        
        # Lưu lộ trình tốt nhất vào file riêng
        best_path_filename = self.filename.replace('.csv', '_best_path.json')
        best_path_output = self.result_path + best_path_filename
        
        # Tách lộ trình thành các route riêng biệt
        routes = []
        current_route = []
        for node in self.best_path:
            if node == 0:
                if current_route:
                    routes.append(current_route)
                current_route = []
            else:
                current_route.append(node)
        
        best_path_data = {
            'best_path': self.best_path,
            'best_path_distance': self.best_path_distance,
            'best_vehicle_num': self.best_vehicle_num,
            'total_time': time.time() - start_time_total,
            'routes': routes,
            'parameters': {
                'ants_num': self.ants_num,
                'max_iter': self.max_iter,
                'alpha': self.alpha,
                'beta': self.beta,
                'q0': self.q0,
                'rho': self.graph.rho
            }
        }
        
        with open(best_path_output, 'w') as f:
            json.dump(best_path_data, f, indent=4)
        print(f'Best path saved to {best_path_filename}')

    def select_next_index(self, ant):
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob = np.power(self.graph.pheromone_mat[current_index][index_to_visit], self.alpha) * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)
        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        N = len(index_to_visit)

        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob

        while True:
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]
