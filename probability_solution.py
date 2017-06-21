"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''

from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function    
    
    A_Node = BayesNode(0,2,name="alarm")
    FA_Node = BayesNode(2,2,name="faulty alarm")
    G_Node = BayesNode(3,2,name="gauge")
    FG_Node = BayesNode(4,2,name="faulty gauge")
    T_Node = BayesNode(5,2,name="temperature")

    # T -> FG
    T_Node.add_child(FG_Node)
    FG_Node.add_parent(T_Node)
    # FG -> G
    FG_Node.add_child(G_Node)
    G_Node.add_parent(FG_Node)
    # T -> G
    T_Node.add_child(G_Node)
    G_Node.add_parent(T_Node)
    # G -> A
    G_Node.add_child(A_Node)
    A_Node.add_parent(G_Node)
    # FA -> A
    FA_Node.add_child(A_Node)
    A_Node.add_parent(FA_Node)

    nodes.append(A_Node)
    nodes.append(FA_Node)
    nodes.append(G_Node)
    nodes.append(FG_Node)
    nodes.append(T_Node)
    
    return BayesNet(nodes)

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    

    '''
    Probability Tables
    ======================================
    P(FA=true) = 0.15
    P(T=true) = 0.2

    FG | T | P(G=true given FG, T)
    ---------------------------------
    T  | T | 0.2
    T  | F | 0.8
    F  | T | 0.95
    F  | F | 0.15

    T | P(FG=true given T)
    --------------------------
    T | 0.8
    F | 0.05

    FA | G | P(A=true given FA, G)
    --------------------------------
    T  | T | 0.55
    T  | F | 0.45
    F  | T | 0.9
    F  | F | 0.1
    '''
    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    # P(FA)
    fa_dist = DiscreteDistribution(F_A_node)
    idx = fa_dist.generate_index([],[])
    fa_dist[idx] = [0.85,0.15]
    F_A_node.set_dist(fa_dist)

    # P(T)
    t_dist = DiscreteDistribution(T_node)
    idx = t_dist.generate_index([],[])
    t_dist[idx] = [0.8,0.2]
    T_node.set_dist(t_dist)

    # P(FG | T)

    return bayes_net

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    # TODO: finish this function
    raise NotImplementedError
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    # TOOD: finish this function
    raise NotImplementedError
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    raise NotImplementedError
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    # TODO: fill this out
    raise NotImplementedError    
    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    raise NotImplementedError
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    
    # TODO: finish this function
    raise NotImplementedError
    return sample

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A= bayes_net.get_node_by_name("A")      
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_state)    
    # TODO: finish this function
    raise NotImplementedError    
    return sample


def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError        
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor