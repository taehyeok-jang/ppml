import numpy as np
import matplotlib.pyplot as plt
import random

# get the pareto-front of data where the first column is the accuracy and the second column is the latency
# no point can have a higher accuracy and lower latency than another point on the pareto-front
def pareto_front_from_model_zoo(data):
    pareto = []
    for i in range(len(data)):
        is_pareto = True
        for j in range(len(data)):
            if (data[j][0] > data[i][0]) and (data[j][1] <= data[i][1]):
                is_pareto = False
                break
        if is_pareto:
            pareto.append(data[i])
    return np.array(sorted(pareto))

def select_element_from_pmf(pmf):
    elements = list(pmf.keys())
    probabilities = list(pmf.values())
    
    # Select an element based on the PMF
    # print(probabilities)
    # print(np.sum(probabilities))
    if np.sum(probabilities)==0:
        return None
    
    selected_element = np.random.choice(elements, p=probabilities)
    
    return selected_element

def median_quantiles(pareto_front):
    acc_median = np.median(np.array(pareto_front)[:,0])
    lat_median = np.median(np.array(pareto_front)[:,1])
    return acc_median, lat_median

# ----- Score Functions -----

def goodput_score(a_served, l_served, a_input, l_input):
    if float(a_served) >= a_input and float(l_served) <= l_input:
        # return 100 if sensitivity is 100
        return 1
    else:
        return 0

# latency distribution may not be the same as the accuracy distribution --> quantile score function solves this
def l1_score(a_served, l_served, a_input, l_input):
        # if a_served < a_input and l_served > l_input:
        #     return 0
        # elif a_served < a_input:
        #     return 1 - (a_input/100 - a_served/100)
        # elif l_served > l_input:
        #     return 1 - (l_served/100 - l_input/100)
        # else:
        #     return 1

        # ***DS give dist-based scores within the feasible set as well instead of just 1 ???
        return (1 - max((a_input/100 - a_served/100),0))/2 + (1 - max((l_served/100 - l_input/100),0))/2


def quantile_score(a_served, l_served, pareto_front):
    acc_median, lat_median = median_quantiles(pareto_front)
    return (1 - max((acc_median/100 - a_served/100),0))/2 + (1 - max((l_served/100 - lat_median/100),0))/2

# ----- Permute and Flip Mechanism functions -----

def goodput_permute_and_flip_mechanism(eps, pareto_front, sensitivity, acc_input, lat_input, reduce_size=False):
    # ***DS implement this -- truncate base set
    # if reduce_size:
    #     old_pareto_front = pareto_front
    #     pass
    
    remaining_indices = np.arange(len(pareto_front))
    np.random.shuffle(remaining_indices)
    while len(remaining_indices) > 0:
        selected_index = remaining_indices[0]
        remaining_indices = remaining_indices[1:]
        point = pareto_front[selected_index]

        optimal_goodput = 1
        point_goodput_score = [goodput_score(point[0], point[1], acc_input, lat_input) - optimal_goodput]
        prob_val = np.exp(eps * np.array(point_goodput_score) / (2 * sensitivity))

        # flip the coin with probability of heads being prob_val and return if it's heads
        if np.random.rand() < prob_val:
            return str(point)

def l1_permute_and_flip_mechanism(eps, pareto_front, sensitivity, acc_input, lat_input, reduce_size=False):
    remaining_indices = np.arange(len(pareto_front))
    np.random.shuffle(remaining_indices)
    while len(remaining_indices) > 0:
        selected_index = remaining_indices[0]
        remaining_indices = remaining_indices[1:]
        point = pareto_front[selected_index]
        # print("print1",point,type(point))

        optimal_goodput = 1
        point_l1_score = [l1_score(point[0], point[1], acc_input, lat_input) - optimal_goodput]
        prob_val = np.exp(eps * np.array(point_l1_score) / (2 * sensitivity))

        # flip the coin with probability of heads being prob_val and return if it's heads
        # if prob_val is large then the chance of random int being less than prob_val is high, so the point is more likely to be selected
        if np.random.rand() < prob_val:
            return str(point)

# ----- Exponential Mechanism functions -----

def non_private_mechanism(pareto_front, acc_input, lat_input):
    feasibility_region = []
    for point in pareto_front:
        if point[0] >= acc_input and point[1] <= lat_input:
            feasibility_region.append([point[0],point[1]])
       
    prob_vals = []
    for point in pareto_front:
        # print(point)
        # print(feasibility_region)
        if [point[0],point[1]] in feasibility_region:
            prob_vals.append(1/len(feasibility_region))
        else:
            prob_vals.append(0)

    pmf = dict()

    for element, probability in zip(pareto_front, prob_vals):
        # print(element, probability)
        pmf[str(element)] = probability
    
    return pmf

def goodput_exponential_mechanism(eps, pareto_front, sensitivity, acc_input, lat_input):
    goodput_scores = []
    for point in pareto_front:
        goodput_scores.append(goodput_score(point[0], point[1], acc_input, lat_input))

    prob_vals = np.exp((eps * np.array(goodput_scores)) / (2 * sensitivity))

    proportional_probs = prob_vals / np.sum(prob_vals)

    pmf = dict()

    for element, probability in zip(pareto_front, proportional_probs):
        # print(element, probability)
        pmf[str(element)] = probability
    
    return pmf

# has to be a probability distribution that is skewed towards the models that are in the feasibility set
# should return a model based on the exponential probability distribution
# there will be a different probability distribution for each (acc, lat) input specification
def l1_exponential_mechanism(eps, pareto_front, sensitivity, acc_input, lat_input):
    l1_scores = []
    for point in pareto_front:
        l1_scores.append(l1_score(point[0], point[1], acc_input, lat_input))
    
    
    # sensitivity_array = np.full((len(pareto_front,)),sensitivity)

    """
    VOL1 
    """
    # prob_vals  = np.exp((eps * np.array(l1_scores)) / (2 * sensitivity))

    # proportional_probs = prob_vals / np.sum(prob_vals)
    ## print(proportional_probs)
    ## print(np.sum(proportional_probs))
    ## print(proportional_probs.shape)

    """
    VOL2
    """
    """
    # Scale the scores
    scaled_scores = (eps * np.array(l1_scores)) / (2 * sensitivity)
    
    # Subtract max for numerical stability
    max_score = np.max(scaled_scores)
    stable_scores = scaled_scores - max_score
    
    # Compute probabilities using stable exponentials
    prob_vals = np.exp(stable_scores)
    proportional_probs = prob_vals / np.sum(prob_vals)
    """

    """
    VOL 3
    """
    max_exp_input = np.log(np.finfo(float).max)
    scaled_scores = (eps * np.array(l1_scores)) / (2 * self.sensitivity)
    scaled_scores = np.clip(scaled_scores, -max_exp_input, max_exp_input)  # Clip to avoid overflow
    prob_vals = np.exp(scaled_scores)

    epsilon = 1e-12
    proportional_probs = prob_vals / (np.sum(prob_vals) + epsilon)

    if np.any(np.isnan(proportional_probs)) or np.sum(proportional_probs) == 0:
        print("Warning: Invalid probabilities detected. Returning None.")
        return None


    pmf = dict()

    for element, probability in zip(pareto_front, proportional_probs):
        # print(element, probability)
        pmf[str(element)] = probability
    
    return pmf

def quantile_exponential_mechanism(eps, pareto_front, sensitivity):
    quantile_scores = []
    for point in pareto_front:
        quantile_scores.append(quantile_score(point[0], point[1], pareto_front))
    
    
    # sensitivity_array = np.full((len(pareto_front,)),sensitivity)
   
    prob_vals  = np.exp((eps * np.array(quantile_scores)) / (2 * sensitivity))

    proportional_probs = prob_vals / np.sum(prob_vals)
    # print(proportional_probs)
    # print(np.sum(proportional_probs))
    # print(proportional_probs.shape)

    pmf = dict()

    for element, probability in zip(pareto_front, proportional_probs):
        # print(element, probability)
        pmf[str(element)] = probability
    
    return pmf

# ----- utility functions -----

# def dist_utility_goodput_score(eps, sensitivity, query_list, pareto_front):

def utility_goodput_score_paf(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    goodput = 0

    for query in query_list:
        selected_element = goodput_permute_and_flip_mechanism(eps, pareto_front, sensitivity, query[0], query[1])
        # we can get none when the query is infeasible, then no point will have a high enough prob to be selected
        if selected_element is not None:
            selected_element = list(selected_element.strip("[]").split()) 
            if float(selected_element[0]) >= query[0] and float(selected_element[1]) <= query[1]:
                goodput += 1

    return goodput/len(query_list)

# goodput based on l1-score-exponential-distribution
def utility_l1_score(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    goodput = 0
    for query in query_list:
        my_pmf = l1_exponential_mechanism(eps, pareto_front, sensitivity, query[0], query[1])

        selected_element = select_element_from_pmf(my_pmf)

        selected_element = list(selected_element.strip("[]").split())
        
        # print(selected_element)
        
        if float(selected_element[0]) >= query[0] and float(selected_element[1]) <= query[1]:
            goodput += 1

    return goodput/len(query_list)

def dist_utility_l1_score(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    utility = 0
    for query in query_list:
        my_pmf = l1_exponential_mechanism(eps, pareto_front, sensitivity, query[0], query[1])

        selected_element = select_element_from_pmf(my_pmf)

        selected_element = list(selected_element.strip("[]").split())
        
        # print(selected_element)
        
        utility += l1_score(float(selected_element[0]), float(selected_element[1]), query[0], query[1])

    return utility/len(query_list)

def dist_utility_l1_score_paf(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    utility = 0
    answered_queries = 0
    for query in query_list:
        selected_element = l1_permute_and_flip_mechanism(eps, pareto_front, sensitivity, query[0], query[1])
        # we can get none when the query is infeasible, then no point will have a high enough prob to be selected
        if selected_element is not None:
            answered_queries += 1
            selected_element = list(selected_element.strip("[]").split())
            utility += l1_score(float(selected_element[0]), float(selected_element[1]), query[0], query[1])

    # return utility/len(query_list)
    return utility/answered_queries

def dist_utility_l1_score_paf_(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    utility = 0
    answered_queries = 0
    selected = [] 
    for query in query_list:
        selected_element = l1_permute_and_flip_mechanism(eps, pareto_front, sensitivity, query[0], query[1])
        # we can get none when the query is infeasible, then no point will have a high enough prob to be selected
        if selected_element is not None:
            answered_queries += 1
            selected_element = list(selected_element.strip("[]").split())
            selected.append(selected_element)
            
            utility += l1_score(float(selected_element[0]), float(selected_element[1]), query[0], query[1])

    # return utility/len(query_list)
    return utility/answered_queries, selected

def utility_goodput_score(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    goodput = 0
    for query in query_list:
        my_pmf = goodput_exponential_mechanism(eps, pareto_front, sensitivity, query[0], query[1])

        selected_element = select_element_from_pmf(my_pmf)

        selected_element = list(selected_element.strip("[]").split())
        
        # print(selected_element)
        
        if float(selected_element[0]) >= query[0] and float(selected_element[1]) <= query[1]:
            goodput += 1

    return goodput/len(query_list)

def utility_quantile_score(eps, sensitivity, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    goodput = 0
    for query in query_list:
        my_pmf = quantile_exponential_mechanism(eps, pareto_front, sensitivity)

        selected_element = select_element_from_pmf(my_pmf)

        selected_element = list(selected_element.strip("[]").split())
        
        # print(selected_element)
        
        if float(selected_element[0]) >= query[0] and float(selected_element[1]) <= query[1]:
            goodput += 1

    return goodput/len(query_list)

def utility_non_private_score(eps, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    goodput = 0
    answered_queries = 0
    for query in query_list:
        my_pmf = non_private_mechanism(pareto_front, query[0], query[1])

        selected_element = select_element_from_pmf(my_pmf)

        if selected_element is None:
            continue

        answered_queries += 1
        selected_element = list(selected_element.strip("[]").split())

        if float(selected_element[0]) >= query[0] and float(selected_element[1]) <= query[1]:
            goodput += 1

    # return goodput/len(query_list)
    return goodput/answered_queries

def dist_utility_non_private_score(eps, query_list, pareto_front):
    print(f"computing goodput for {eps}")
    utility = 0
    answered_queries = 0
    for query in query_list:
        my_pmf = non_private_mechanism(pareto_front, query[0], query[1])

        selected_element = select_element_from_pmf(my_pmf)

        if selected_element is None:
            continue

        answered_queries += 1
        selected_element = list(selected_element.strip("[]").split())

        utility += l1_score(float(selected_element[0]), float(selected_element[1]), query[0], query[1])

    return utility/answered_queries

# ----- Plotting functions -----

# plot the pareto-front as scatter and join the points with a line
def plot_pareto_front(pareto_front):
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_front[:, 1], pareto_front[:, 0], label='pareto-front', color='r')
    plt.plot(pareto_front[:, 1], pareto_front[:, 0])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Accuracy')
    plt.legend()

# plot for epsilon-utility with epsilon values
def plot_utility(epsilons, utility_vals, mechanism):
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, utility_vals)
    plt.scatter(epsilons, utility_vals)
    plt.xscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Utility')
    plt.xticks(epsilons, [str(i) for i in epsilons])
    plt.title(f'Epsilon-Utility for {mechanism.__name__}')
    plt.grid(True)
    plt.show()

# plot pmf for a given query
def plot_pmf(eps, sensitivity, query, pareto_front, mechanism):
    plt.figure(figsize=(16, 6))
    if mechanism == l1_exponential_mechanism:
        my_pmf = mechanism(eps, pareto_front, sensitivity, query[0], query[1])
        plt.title(f'PMF for query {query}, eps={eps}, sensitivity={sensitivity}, mechanism = {mechanism.__name__}')
    elif mechanism == non_private_mechanism:
        my_pmf = mechanism(pareto_front, query[0], query[1])
        plt.title(f'PMF for query {query},  mechanism = {mechanism.__name__}')
    elif mechanism == goodput_exponential_mechanism:
        my_pmf = mechanism(eps, pareto_front, sensitivity, query[0], query[1])
        plt.title(f'PMF for query {query}, eps={eps}, sensitivity={sensitivity}, mechanism = {mechanism.__name__}')
    else:
        my_pmf = mechanism(eps, pareto_front, sensitivity)
        plt.title(f'PMF for query {query}, eps={eps}, sensitivity={sensitivity}, acc_median = 89.5; lat_median = 52.5,  mechanism = {mechanism.__name__}')
    x = list(my_pmf.keys())
    y = list(my_pmf.values())
    plt.bar(x, y)
    plt.xlabel('Model')
    plt.ylabel('Probability')
    plt.show()