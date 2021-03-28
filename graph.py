import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import njit, prange
from numba.typed import Dict, List
from numba.types import int64, ListType 
from numba.experimental import jitclass
from numba import typeof
import _hits

import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, describe


ITERATIONS = 100
NUMBER_OF_AGENTS = 20
ROUNDS = 100
SAVE_IMG = False
# MC = 500
# SIGNIFICANCE = 1e-2

@jitclass([('agents', ListType(int64)),('current_nodes', ListType(int64))])
class Chat:
    def __init__(self, agents, cn):
        self.agents = agents # defines write rights to chat
        self.current_nodes = cn


@jitclass([('chats', ListType(int64))])
class Agent:
    def __init__(self, group_ids):
        self.chats = group_ids # defines read rights to chat
        # self.reachable_nodes = List.empty_list(int64)





def fill_chats_and_agents(chats, agents, agents_in_chats):
    # initalise agents
    for a in range(NUMBER_OF_AGENTS):
        agents.append(Agent(List.empty_list(int64)))
    # fill chats and agents
    for chat_id in range(len(agents_in_chats)):
        chat = agents_in_chats[chat_id]
        chats.append(Chat(List(chat), List([0])))
        for agent_id in chat:
            agents[agent_id].chats.append(chat_id)

    return chats, agents

@njit
def get_active_agents(agents):
    result = List.empty_list(int64)
    for agent_id in prange(len(agents)):
        if np.random.rand() < 0.2:
            result.append(agent_id) # key is agent id
    return result

def generate(chats, agents):
    result = {}
    for iround in range (ROUNDS):
        result[iround] = []
        agents_active = get_active_agents(agents)
        for i in range(len(agents_active)):
            agent_id = agents_active[i]
            chat_id = np.random.choice(agents[agent_id].chats)
            while not agent_id in chats[chat_id].agents:
                chat_id = np.random.choice(agents[agent_id].chats)
            result[iround].append((agent_id, chat_id))

    return result

@njit
def add_node_to_graph(agent_id, node_id, chats, agents, graph, graph_transitive):
        # find all possible references of "other" chats the agent has access to
        references = List.empty_list(int64)
        for other_c in prange(len(agents[agent_id].chats)):
            chat_id = agents[agent_id].chats[other_c]
            chat_current_nodes = chats[chat_id].current_nodes
            for cn in chat_current_nodes:
                if not cn in references:
                    references.append(cn)

        # exclude emtpy chats
        if 0 in references and len(references) > 1:
            references = List(filter(lambda a: a != 0, references))

        # exclude transitive refs
        cp_ref = references.copy()
        len_ref = len(cp_ref)
        for r in range(len_ref):
            ref = cp_ref[r]
            trefs = graph_transitive.get(ref)
            for t in trefs:
                if t in references:
                    references.remove(t)
        graph[node_id] = references 

        references = references.copy()
        len_ref = len(references)
        for r in range(len_ref):
            ref = references[r]
            trefs = graph_transitive.get(ref)
            for t in trefs:
                if not t in references:
                    references.append(t)
        graph_transitive[node_id] = references



# @njit(nogil=True)
def run(game, chats, agents, graph, graph_transitive, node_chat, node_agent):
    for iround in range(ROUNDS):
        base_node_id = len(graph) # then just + chad_id gives unique id
        # agents_active = get_active_agents(agents)
        agents_active = game[iround]
        # # print(agents_active)
        chatter = List.empty_list(int64)
        # add a node
        for i in prange(len(agents_active)):
            agent_id = agents_active[i][0]
            node_id = base_node_id + i
            chat_id = agents_active[i][1] # np.random.choice(agents[agent_id].chats)
            # while not agent_id in chats[chat_id].agents:
            #     chat_id = np.random.choice(agents[agent_id].chats)
            # # print("\t", agent_id, chat_id, node_id)
            chatter.append(chat_id)
            node_chat[node_id] = chat_id
            node_agent[node_id] = agent_id
            add_node_to_graph(agent_id, node_id, chats, agents, graph, graph_transitive)

        # important to do the chat update after graph
        for i in prange(len(chatter)):
            chat_id = chatter[i]
            node_id = base_node_id + i
            for cn in chats[chat_id].current_nodes:
                if cn < base_node_id:
                    chats[chat_id].current_nodes = List.empty_list(int64)
                    break
            chats[chat_id].current_nodes.append(node_id)




@njit(nogil=True)
def to_adj_matrix(graph,M):
    for i in prange(len(graph)):
        for k in prange(len(graph[i])):
            j = graph[i][k]
            M[i,j] = True



def run_game(game, chats, agents, with_app_links=True, conf=True):
    global c_bl_n, c_sl_n, t_bl_n, t_sl_n, c_bl_nn, c_sl_nn, t_bl_nn, t_sl_nn
    # print("App Links:",with_app_links)
    # print()
    for chat in chats:
        chat.current_nodes = List([0])
    graph = Dict.empty(int64, ListType(int64))
    graph[0] = List.empty_list(int64)
    graph_transitive = Dict.empty(int64, ListType(int64))
    graph_transitive[0] = List.empty_list(int64)
    node_chat = Dict.empty(int64, int64)
    node_chat[0] = -1
    node_agent = Dict.empty(int64, int64)
    node_agent[0] = -1

    run(game, chats, agents, graph, graph_transitive, node_chat, node_agent)
    M = np.zeros((len(graph),len(graph)), dtype=bool)
    to_adj_matrix(graph,M)

    # add APP links
    if with_app_links:
        chats_last_node = np.zeros(len(chats))
        for n in range(M.shape[0]):
            last_node_of_chat = int(chats_last_node[node_chat[n]])
            if last_node_of_chat != 0:
                M[n, last_node_of_chat] = True
            chats_last_node[node_chat[n]] = n

    try: 
        # print("Normalized")
        data_n = break_target(M.copy(), node_agent, agents)
        # print()
        # print("Not Normalized")
        data_nn = break_target(M.copy(), node_agent, agents, norm=False)
        if conf:
            if with_app_links:
                c_bl_n = c_bl_n.append(data_n, ignore_index=True)
                c_bl_nn = c_bl_nn.append(data_nn, ignore_index=True)
            else:
                c_sl_n = c_sl_n.append(data_n, ignore_index=True)
                c_sl_nn = c_sl_nn.append(data_nn, ignore_index=True)
        else:
            if with_app_links:
                t_bl_n = t_bl_n.append(data_n, ignore_index=True)
                t_bl_nn = t_bl_nn.append(data_nn, ignore_index=True)
            else:
                t_sl_n = t_sl_n.append(data_n, ignore_index=True)
                t_sl_nn = t_sl_nn.append(data_nn, ignore_index=True)
    except Exception as err:
        print(err)
    # print()
    return M

def add_plot(data):
    plt.figure()
    plt.hist(data, density=True, bins=10)  # density=False would make counts
    plt.ylim(0,data.shape[0]//2)
    plt.xlim(0,1)
    plt.ylabel('Frequency')
    plt.xlabel('Score')


def break_target(A, node_agent, agents, norm=True):
    # print(A.shape)
    k = 1
    M = A.copy()
    G_n = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    node_count = len(G_n.nodes())
    edge_count = len(G_n.edges())
    in_deg = np.mean(list(map(lambda x: x[1],G_n.in_degree())))
    # print("In_Deg:\t",in_deg)
    nodes = list(G_n.nodes())
    
    new_node = len(nodes)
    index = new_node
    for agent_id in range(len(agents)):
        G_n.add_node(index+agent_id)
        for node in nodes:
            if G_n.in_degree(node) == 0  and node_agent[node] == agent_id:
                G_n.add_edge(index+agent_id, node)
    G_t = nx.transitive_closure(G_n)
    M_t = nx.to_numpy_array(G_t, dtype=np.float64)
    auth_base = _hits.hits_authorities(M_t, normalized=norm)
    if SAVE_IMG:
        add_plot(auth_base)
    base_sum = np.sum(auth_base)
    base_median = np.median(auth_base)
    base_mean = np.mean(auth_base)
    # print("Sum:\t",base_sum)
    # print("Median:\t",base_median)
    # print("Mean:\t",base_mean)
    G_n = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    G_t = nx.transitive_closure(G_n)
    M_t = nx.to_numpy_array(G_t, dtype=np.float64)
    while True:
        G_n = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
        stats = nx.betweenness_centrality(G_n)  
        node = max(stats, key=lambda key: stats[key])
        M = np.delete(M, node, axis=0)
        M = np.delete(M, node, axis=1)
        G_n.remove_node(node)
        if not nx.is_weakly_connected(G_n):
            index = new_node
            nodes = list(G_n.nodes())
            for agent_id in range(len(agents)):
                G_n.add_node(index+agent_id)    
                for node in nodes:
                    if G_n.in_degree(node) == 0 and node_agent[node] == agent_id:
                        G_n.add_edge(index+agent_id, node)
            G_t = nx.transitive_closure(G_n)
            M_t = nx.to_numpy_array(G_t, dtype=np.float64)
            auth_dc = _hits.hits_authorities(M_t, normalized=norm)        
            # print("Disconnectedness")
            dc_sum = np.sum(auth_dc)
            dc_median = np.median(auth_dc)
            dc_mean = np.mean(auth_dc)
            # print("Sum:\t",dc_sum)
            # print("Median:\t",dc_median)
            # print("Mean:\t",dc_mean)
            if SAVE_IMG:
                add_plot(auth_dc)
                plt.show()
            d = np.array([node_count,edge_count,base_sum,base_median,base_mean,dc_sum,dc_median,dc_mean,in_deg,k])
            return pd.Series(data=d, index=['Nodes', 'Edges', 'Sum (Base)', 'Median (Base)', 'Mean (Base)', 'Sum (DC)', 'Median (DC)', 'Mean (DC)', 'IN_DEG', 'DISCON'])    
            # return k
        k += 1
    return -1





# c confidential
# bl both links
# n normalized
cols = ['Nodes', 'Edges', 'Sum (Base)', 'Median (Base)', 'Mean (Base)', 'Sum (DC)', 'Median (DC)', 'Mean (DC)', 'IN_DEG', 'DISCON']
c_bl_n = pd.DataFrame(columns=cols)
c_bl_nn = pd.DataFrame(columns=cols)
c_sl_n = pd.DataFrame(columns=cols)
c_sl_nn = pd.DataFrame(columns=cols)
t_bl_n = pd.DataFrame(columns=cols)
t_bl_nn = pd.DataFrame(columns=cols)
t_sl_n = pd.DataFrame(columns=cols)
t_sl_nn = pd.DataFrame(columns=cols)





for its in range(ITERATIONS):

    # # generate agent graph for CHATS OF TWOS
    # beta model
    G_a = nx.generators.random_graphs.connected_watts_strogatz_graph(NUMBER_OF_AGENTS,4,0.25)
    # # random model
    # while True:
    #     G_a = nx.gnp_random_graph(NUMBER_OF_AGENTS,0.4)
    #     if nx.is_connected(G_a):
    #         break
    # # fill in agents_in_chats
    agents_in_chats = []
    # fill in from graph for CHATS OF TWOS
    agents_in_chats = list(map(list,list(G_a.edges())))
    # # fill in sample random for CHATS OF VARIOUS # timeline style
    for n in G_a.nodes():
        agents_in_chats.append(list(G_a.neighbors(n)))




    a = Agent(List.empty_list(int64))
    agents = List.empty_list(typeof(a))

    c = Chat(List.empty_list(int64), List([0]))
    chats = List.empty_list(typeof(c))


    chats, agents = fill_chats_and_agents(chats,agents, agents_in_chats)

    ##############
    #    Game    #
    ##############
    game = generate(chats, agents)



    # confidential
    # print("CONFIDENTIAL")
    # print()
    run_game(game, chats, agents, with_app_links=True, conf=True)
        
    # print()
    run_game(game, chats, agents, with_app_links=False, conf=True)
    # print()


    # print()
    # defines read rights to chat: full transparency
    for agent in agents:
        agent.chats = List(range(len(chats)))

    # transparent
    # print("TRANSPARENT")
    # print()
    run_game(game, chats, agents, with_app_links=True, conf=False)
    # print()
    run_game(game, chats, agents, with_app_links=False, conf=False)
    # print()
    print(f"{its+1} done.")





print(c_bl_n.shape)
# print("c_bl_n")
# print(c_bl_n.mean(axis=0))
ds = c_bl_n.mean(axis=0)
ds.name = "c_bl_n"
ds.to_pickle(f"./data/c_bl_n.pkl") 
# print()
# print("c_bl_nn")
# print(c_bl_nn.mean(axis=0))
ds = c_bl_nn.mean(axis=0)
ds.name = "c_bl_nn"
ds.to_pickle(f"./data/c_bl_nn.pkl") 
# print()
# print("c_sl_n")
# print(c_sl_n.mean(axis=0))
ds = c_sl_n.mean(axis=0)
ds.name = "c_sl_n"
ds.to_pickle(f"./data/c_sl_n.pkl") 
# print()
# print("c_sl_nn")
# print(c_sl_nn.mean(axis=0))
ds = c_sl_nn.mean(axis=0)
ds.name = "c_sl_nn"
ds.to_pickle(f"./data/c_sl_nn.pkl") 
# print()
# print("t_bl_n")
# print(t_bl_n.mean(axis=0))
ds = t_bl_n.mean(axis=0)
ds.name = "t_bl_n"
ds.to_pickle(f"./data/t_bl_n.pkl") 
# print()
# print("t_bl_nn")
# print(t_bl_nn.mean(axis=0))
ds = t_bl_nn.mean(axis=0)
ds.name = "t_bl_nn"
ds.to_pickle(f"./data/t_bl_nn.pkl") 
# print()
# print("t_sl_n")
# print(t_sl_n.mean(axis=0))
ds = t_sl_n.mean(axis=0)
ds.name = "t_sl_n"
ds.to_pickle(f"./data/t_sl_n.pkl") 
# print()
# print("t_sl_nn")
# print(t_sl_nn.mean(axis=0))
ds = t_sl_nn.mean(axis=0)
ds.name = "t_sl_nn"
ds.to_pickle(f"./data/t_sl_nn.pkl") 
# print()




print()
print(pd.read_pickle(f"./data/c_bl_n.pkl"))

print()
print(pd.read_pickle(f"./data/c_bl_nn.pkl"))

print()
print(pd.read_pickle(f"./data/c_sl_n.pkl"))

print()
print(pd.read_pickle(f"./data/c_sl_nn.pkl"))

print()
print(pd.read_pickle(f"./data/t_bl_n.pkl"))

print()
print(pd.read_pickle(f"./data/t_bl_nn.pkl"))

print()
print(pd.read_pickle(f"./data/t_sl_n.pkl"))

print()
print(pd.read_pickle(f"./data/t_sl_nn.pkl"))