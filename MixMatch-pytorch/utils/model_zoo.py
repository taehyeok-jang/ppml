import numpy as np 
import torch

class ModelZooServer:
    def __init__(self, model_zoo, pareto_front, eps, sensitivity):
        self.model_zoo = model_zoo
        self.pareto_front = np.array(pareto_front) 
        self.eps = eps
        self.sensitivity = sensitivity

    
    def l1_permute_and_flip_mechanism(query_point):
        remaining_indices = np.arange(len(self.pareto_front))
        np.random.shuffle(remaining_indices)
        while len(remaining_indices) > 0:
            selected_index = remaining_indices[0]
            remaining_indices = remaining_indices[1:]
            point = self.pareto_front[selected_index]

            optimal_goodput = 1
            point_l1_score = [l1_score(point[0], point[1], query_point[0], query_point[1]) - optimal_goodput]
            prob_val = np.exp(self.eps * np.array(point_l1_score) / (2 * self.sensitivity))

            # flip the coin with probability of heads being prob_val and return if it's heads
            # if prob_val is large then the chance of random int being less than prob_val is high, so the point is more likely to be selected
            if np.random.rand() < prob_val:
                return str(point)
            
    def l1_score(a_served, l_served, a_input, l_input):
        
        return (1 - max((a_input/100 - a_served/100),0))/2 + (1 - max((l_served/100 - l_input/100),0))/2

    def query(self, query_point):
        selected_index = self.l1_permute_and_flip_mechanism(query_point)
        selected_model_name = list(self.model_zoo.keys())[selected_index]
        
        return self.model_zoo[selected_model_name]