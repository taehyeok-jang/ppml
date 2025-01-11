import numpy as np 
import torch

class PsmlDefenseProxy:
    def __init__(self, pareto_front_models, pareto_front_spec, eps, sensitivity):
        """
        Args:
            # model_zoo: 
            #     (e.g.) {
            #         'mobilenetv2_x0_5': {'accuracy': 71.04, 'latency': 36.91},
            #         ... 
            #         'vgg16_bn': {'accuracy': 74.68, 'latency': 16.6},
            #         'vit_base_patch16_384': {'accuracy': 90.1, 'latency': 39.29},
            #         'vit_large_patch16_384': {'accuracy': 90.82, 'latency': 73.69}}
            #     }

            pareto_front_models: 
                (e.g.) [
                    'vgg11_bn',
                    'vgg13_bn',
                    'vit_large_patch16_384',
                ]

            pareto_front_spec: the specification of models on pareto front; 
                (e.g.) [
                    [93.22, 11.07], 
                    [93.72, 12.83],
                    [98.38, 67.79]
                ]

            eps: 

            sensitivity: 

        """
        print("Initializing PsmlDefenseProxy... w/ eps {}, sensitivity {}".format(eps, sensitivity))
        '''
        for model, spec in zip(pareto_front_models, pareto_front_spec):
            print(model, spec)
        '''

        self.pareto_front = np.array(pareto_front_spec) 
        self.eps = eps
        self.sensitivity = sensitivity

        self.pf_size = len(pareto_front_models)
        self.pareto_front_map = {tuple(pareto_front_spec[i]): pareto_front_models[i] for i in range(self.pf_size)}

    
    def l1_permute_and_flip_mechanism(self, query_point):
        remaining_indices = np.arange(self.pf_size)
        np.random.shuffle(remaining_indices)
        while len(remaining_indices) > 0:
            selected_index = remaining_indices[0]
            remaining_indices = remaining_indices[1:]
            point = self.pareto_front[selected_index]

            optimal_goodput = 1
            point_l1_score = [self.l1_score(point[0], point[1], query_point[0], query_point[1]) - optimal_goodput]
            prob_val = np.exp(self.eps * np.array(point_l1_score) / (2 * self.sensitivity))

            # flip the coin with probability of heads being prob_val and return if it's heads
            # if prob_val is large then the chance of random int being less than prob_val is high, so the point is more likely to be selected
            if np.random.rand() < prob_val:
                return tuple(point)
            
    def l1_score(self, a_served, l_served, a_input, l_input):
        
        return (1 - max((a_input/100 - a_served/100),0))/2 + (1 - max((l_served/100 - l_input/100),0))/2

    
    def m_query(self, query_point):
        """
        Given a query_point (accuracy, latency) in the form of a tuple (e.g., (93.92, 15.14)),
        return the corresponding model_name that precisely matches the specifications.

        Args:
            query_point (tuple): A tuple representing the model's accuracy and latency.

        Returns:
            str: The name of the model that matches the given specifications.
        """
        if query_point in self.pareto_front_map:
            model_name = self.pareto_front_map[query_point]
            return model_name
            
        raise ValueError(f"Model {model_name} not available.")
