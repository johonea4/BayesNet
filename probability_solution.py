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
from numpy import zeros, float32, random
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine
from Inference import EnumerationEngine
from copy import deepcopy

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
    fg_dist = zeros([T_node.size(),F_G_node.size()],dtype=float32)
    fg_dist[0,:] = [0.95,0.05]
    fg_dist[1,:] = [0.2,0.8]
    fg = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node],table=fg_dist)
    F_G_node.set_dist(fg)

    # P(G | FG,T)
    g_dist = zeros([F_G_node.size(),T_node.size(),G_node.size()],dtype=float32)
    g_dist[0,0,:] = [0.95,0.05]
    g_dist[0,1,:] = [0.05,0.95]
    g_dist[1,0,:] = [0.2,0.8]
    g_dist[1,1,:] = [0.8,0.2]
    g=ConditionalDiscreteDistribution(nodes=[F_G_node,T_node,G_node],table=g_dist)
    G_node.set_dist(g)

    # P(A | FA,G)
    a_dist = zeros([F_A_node.size(),G_node.size(),A_node.size()],dtype=float32)
    a_dist[0,0,:] = [0.9,0.1]
    a_dist[0,1,:] = [0.1,0.9]
    a_dist[1,0,:] = [0.55,0.45]
    a_dist[1,1,:] = [0.45,0.55]
    a=ConditionalDiscreteDistribution(nodes=[F_A_node,G_node,A_node],table=a_dist)
    A_node.set_dist(a)

    return bayes_net

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""

    A_node = bayes_net.get_node_by_name("alarm")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    idx = Q.generate_index([alarm_rings],range(Q.nDims))

    alarm_prob = Q[idx]

    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""

    G_node = bayes_net.get_node_by_name("gauge")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    idx = Q.generate_index([gauge_hot],range(Q.nDims))

    gauge_prob = Q[idx]  
    
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""

    # P(T=true | A=true, FA=false, FG=false)
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    idx = Q.generate_index([temp_hot],range(Q.nDims))

    temp_prob = Q[idx]

    return temp_prob

'''
Game Probabilities
==============================

SL | P(SL) where SL=A, B, or C
------------------------------
0  | 0.15
1  | 0.45
2  | 0.3
3  | 0.1

A-B | P(AvB=0 | A-B) | P(AvB=1 | A-B) | P(AvB=2 | A-B)
-------------------------------------------------------
0   | 0.1            | 0.1            | 0.8
1   | 0.2            | 0.6            | 0.2
2   | 0.15           | 0.75           | 0.1
3   | 0.05           | 0.9            | 0.05

'''
def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    
    A = BayesNode(0,4,name="A")
    B = BayesNode(1,4,name="B")
    C = BayesNode(2,4,name="C")
    AvB = BayesNode(3,3,name="AvB")
    BvC = BayesNode(4,3,name="BvC")
    CvA = BayesNode(5,3,name="CvA")

    # A -> AvB / A -> CvA
    A.add_child(AvB)
    A.add_child(CvA)
    AvB.add_parent(A)
    CvA.add_parent(A)

    # B -> AvB / B -> BvC
    B.add_child(AvB)
    B.add_child(BvC)
    AvB.add_parent(B)
    BvC.add_parent(B)

    # C -> BvC / C -> CvA
    C.add_child(BvC)
    C.add_child(CvA)
    BvC.add_parent(C)
    CvA.add_parent(C)

    N = [A,B,C]
    for n in N:
        dist = DiscreteDistribution(n)
        idx = dist.generate_index([],[])
        dist[idx] = [0.15,0.45,0.3,0.1]
        n.set_dist(dist)

    N = [AvB,BvC,CvA]
    NN = [ [A,B],[B,C],[C,A] ]
    c=0
    for n in N:
        dist = zeros([NN[c][0].size(),NN[c][1].size(),n.size()],dtype=float32)
        for i in range(0,4):
            for j in range(0,4):
                if( (j-i) == 0):
                    dist[i,j,:] = [0.1,0.1,0.8]
                elif ((j-i)==1):
                    dist[i,j,:] = [0.2,0.6,0.2]
                elif ((j-i)==2):
                    dist[i,j,:] = [0.15,0.75,0.1]
                elif ((j-i)==3):
                    dist[i,j,:] = [0.05,0.9,0.05]
                elif ((j-i)==-1):
                    dist[i,j,:] = [0.6,0.2,0.2]
                elif ((j-i)==-2):
                    dist[i,j,:] = [0.75,0.15,0.1]
                elif ((j-i)==-3):
                    dist[i,j,:] = [0.9,0.05,0.05]

        tmp = ConditionalDiscreteDistribution(nodes=[NN[c][0],NN[c][1],n],table=dist)
        n.set_dist(tmp)
        c += 1
    
    nodes = [ A,B,C,AvB,BvC,CvA ]

    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    AvB = bayes_net.get_node_by_name("AvB")
    BvC = bayes_net.get_node_by_name("BvC")
    CvA = bayes_net.get_node_by_name("CvA")

    engine = EnumerationEngine(bayes_net)
    engine.evidence[AvB] = 0
    engine.evidence[CvA] = 2

    Q = engine.marginal(BvC)[0]
    idx1 = Q.generate_index(0,range(Q.nDims))
    idx2 = Q.generate_index(1,range(Q.nDims))
    idx3 = Q.generate_index(2,range(Q.nDims))
    
    posterior = [Q[idx1],Q[idx2],Q[idx3]]
    
    return posterior # list 

def calculateMB(node, nodelist, sample):
    states=dict()
    for i in range(0,len(sample)):
        states[nodelist[i]] = sample[i]

    children = node.children
    currentState = states[node]
    P_n = DiscreteDistribution(node)
    samp = []
    normal = 0

    for i in range(node.size()):
        states[node]=i
        pidx = P_n.generate_index(i,range(P_n.nDims))
        dist_nodes = node.dist.nodes
        v = []
        for d in dist_nodes:
            v.append(states[d])
        idx = node.dist.generate_index(v,range(node.dist.nDims))
        P_n[pidx] = node.dist[idx]
        for c in children:
            cdist_nodes = c.dist.nodes
            cv = []
            for d in cdist_nodes:
                cv.append(states[d])
            idx = c.dist.generate_index(cv,range(c.dist.nDims))
            P_n[pidx] *= c.dist[idx]
        samp.append(P_n[pidx])
        normal += P_n[pidx]

    states[node]=currentState
    for i in range(len(samp)):
        samp[i] /= normal
    
    return samp

#Gibbs Implementation. The function sample_value_given_mb was used as a guide
#for performing the markov blanket calculation and utilizing methods in BayesNode,
#Distribution, etc.
def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    #Get The Nodes
    A = bayes_net.get_node_by_name("A")
    B = bayes_net.get_node_by_name("B")
    C = bayes_net.get_node_by_name("C")
    AvB = bayes_net.get_node_by_name("AvB")
    BvC = bayes_net.get_node_by_name("BvC")
    CvA = bayes_net.get_node_by_name("CvA")
    
    nodelist = [A,B,C,AvB,BvC,CvA]
    states = dict()

    #See if inititial_state is specified
    if initial_state == None or len(initial_state) != 6:
        initial_state = [0,0,0,0,0,0]
        initial_state[0] = random.randint(0,4)
        initial_state[1] = random.randint(0,4)
        initial_state[2] = random.randint(0,4)
        initial_state[3] = random.randint(0,3)
        initial_state[4] = random.randint(0,3)
        initial_state[5] = random.randint(0,3)

    sample = list(initial_state)

    ridx = random.randint(0,6)
    n = nodelist[ridx]

    samp = calculateMB(n,nodelist,sample)
    val = random.choice(range(n.size()),p=samp)
    sample[ridx] = val

    return tuple(sample)

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A = bayes_net.get_node_by_name("A")
    B = bayes_net.get_node_by_name("B")
    C = bayes_net.get_node_by_name("C")
    AvB = bayes_net.get_node_by_name("AvB")
    BvC = bayes_net.get_node_by_name("BvC")
    CvA = bayes_net.get_node_by_name("CvA")

    A_dist = A.dist.table
    B_dist = B.dist.table
    C_dist = C.dist.table
    AvB_dist = AvB.dist.table
    BvC_dist = BvC.dist.table
    CvA_dist = CvA.dist.table
  
        #See if inititial_state is specified
    if initial_state == None or len(initial_state) != 6:
        initial_state = [0,0,0,0,0,0]
        initial_state[0] = random.randint(0,4)
        initial_state[1] = random.randint(0,4)
        initial_state[2] = random.randint(0,4)
        initial_state[3] = random.randint(0,3)
        initial_state[4] = random.randint(0,3)
        initial_state[5] = random.randint(0,3)

    sample = list(initial_state)
    nodeList = [A,B,C,AvB,BvC,CvA]
    
    X0 = A_dist[sample[0]] * B_dist[sample[1]] * C_dist[sample[2]]
    X0 *= AvB_dist[sample[0]][sample[1]][sample[3]] * BvC_dist[sample[1]][sample[2]][sample[4]] * CvA_dist[sample[2]][sample[0]][sample[5]]

    oState = deepcopy(sample)

    for i in range(0,6):
        sample[i] = random.randint(0,nodeList[i].size())

    XC = A_dist[sample[0]] * B_dist[sample[1]] * C_dist[sample[2]]
    XC *= AvB_dist[sample[0]][sample[1]][sample[3]] * BvC_dist[sample[1]][sample[2]][sample[4]] * CvA_dist[sample[2]][sample[0]][sample[5]]

    if(XC > X0):
        return tuple(sample)
    else:
        XCp = XC/X0
        u = random.random_sample()
        if(XCp > u):
            return tuple(sample)
        else:
            return tuple(oState)


def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    N=20
    MinLoop=100

    if initial_state == None or len(initial_state) != 6:
        initial_state = [0,0,0,0,0,0]
        initial_state[0] = random.randint(0,4)
        initial_state[1] = random.randint(0,4)
        initial_state[2] = random.randint(0,4)
        initial_state[3] = 0
        initial_state[4] = random.randint(0,3)
        initial_state[5] = 2
    else:
        initial_state[3]=0
        initial_state[5]=2
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH

    gibbsDone=False
    mhDone=False
    gibbSuccessive=0
    mhSuccessive=0
    goState = deepcopy(initial_state)
    gnState = []
    moState = deepcopy(initial_state)
    mnState = []

    while gibbsDone==False or mhDone==False:
        if gibbsDone == False:
            gnState = Gibbs_sampler(bayes_net,goState)
            if(gnState[3]==0 and gnState[5]==2):
                Gibbs_count+=1
                goState = gnState
                lastRun = deepcopy(Gibbs_convergence)
                Gibbs_convergence[gnState[4]]+=1
                if(Gibbs_count>MinLoop):
                    diff = []
                    for i in range(len(Gibbs_convergence)):
                        v = float(Gibbs_convergence[i])/float(Gibbs_count)
                        v-= float(lastRun[i])/float(Gibbs_count-1)
                        diff.append(abs(v))
                    diffLess=True
                    for i in range(len(diff)):
                        diffLess &= diff[i] <= delta
                    if diffLess:
                        gibbSuccessive+=1
                    else:
                        gibbSuccessive=0
                    if gibbSuccessive >= N:
                        gibbsDone=True
        if mhDone == False:
            mnState = MH_sampler(bayes_net,moState)
            if(mnState[3]==0 and mnState[5]==2):
                if(mnState == moState):
                    MH_rejection_count += 1
                else:
                    MH_count += 1
                    moState = mnState
                    lastRun = deepcopy(MH_convergence)
                    MH_convergence[mnState[4]] += 1
                    if(MH_count>MinLoop):
                        diff = []
                        for i in range(len(MH_convergence)):
                            v = float(MH_convergence[i])/float(MH_count)
                            v-= float(lastRun[i])/float(MH_count-1)
                            diff.append(abs(v))
                        diffLess=True
                        for i in range(len(diff)):
                            diffLess &= diff[i] <= delta
                        if diffLess:
                            mhSuccessive+=1
                        else:
                            mhSuccessive=0
                        if mhSuccessive >= N:
                            mhDone=True



    for i in range(len(Gibbs_convergence)):
        Gibbs_convergence[i] /= float(Gibbs_count)
    for i in range(len(MH_convergence)):
        MH_convergence[i] /= float(MH_count)

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor