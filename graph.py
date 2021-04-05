import multiprocessing as mp
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

CORES = 14
ITERATIONS = 1000
NUMBER_OF_AGENTS = 50
ROUNDS = 100
ACTIVATION = 0.1
# SAVE_IMG = False
# MC = 500
# SIGNIFICANCE = 1e-2


@jitclass([('agents', ListType(int64)), ('current_nodes', ListType(int64))])
class Chat:
    def __init__(self, agents, cn):
        self.agents = agents  # defines write rights to chat
        self.current_nodes = cn


@jitclass([('chats', ListType(int64))])
class Agent:
    def __init__(self, group_ids):
        self.chats = group_ids  # defines read rights to chat
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
        if np.random.rand() < ACTIVATION:
            result.append(agent_id)  # key is agent id
    return result


def generate(chats, agents, rounds=ROUNDS):
    result = {}
    for iround in range(rounds):
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
            # if cn >= node_id:
                # print(node_id, "must not reference", cn, "from chat", chat_id, " : ", max(graph, key=int))
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
        if trefs is None:
            graph_transitive[ref] = List.empty_list(int64)
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
def run(game, chats, agents, graph, graph_transitive, node_chat, node_agent, rounds=ROUNDS):
    for iround in range(rounds):
        base_node_id = max(graph, key=int) + 1 # len(graph)  # then just + chad_id gives unique id
        # agents_active = get_active_agents(agents)
        agents_active = game[iround]
        # # print(agents_active)
        chatter = List.empty_list(int64)
        # add a node
        for i in prange(len(agents_active)):
            agent_id = agents_active[i][0]
            node_id = base_node_id + i
            # np.random.choice(agents[agent_id].chats)
            chat_id = agents_active[i][1]
            # while not agent_id in chats[chat_id].agents:
            #     chat_id = np.random.choice(agents[agent_id].chats)
            # # print("\t", agent_id, chat_id, node_id)
            chatter.append(chat_id)
            node_chat[node_id] = chat_id
            node_agent[node_id] = agent_id
            add_node_to_graph(agent_id, node_id, chats,
                              agents, graph, graph_transitive)

        # important to do the chat update after graph
        for i in prange(len(chatter)):
            chat_id = chatter[i]
            node_id = base_node_id + i
            for cn in chats[chat_id].current_nodes:
                if cn < base_node_id:
                    chats[chat_id].current_nodes = List.empty_list(int64)
                    break
            chats[chat_id].current_nodes.append(node_id)

# @njit(nogil=True)
def to_adj_matrix(graph, M):
    for i in graph.keys():
        # if len(graph[i]) == 0:
            # continue
        for j in graph[i]: # k as index
            # j = graph[i][k]
            if j >= M.shape[1]:
                print(M.shape,j)
                print()
                print(graph)
                print(M)
            M[i, j] = True

# to adj list
def from_adj_matrix(G):
    result = Dict.empty(int64, ListType(int64))
    for edge in G.edges():
        if result.get(edge[0]) is None:
            result[edge[0]] = List.empty_list(int64)
        result[edge[0]].append(edge[1])
    return result


def run_game(game, chats, agents, with_app_links=True, conf=True, no_score_links=False):
    global c_bl_n_auth_base, c_bl_n_auth_dc, c_sl_n_auth_base, c_sl_n_auth_dc, t_bl_n_auth_base, t_bl_n_auth_dc, t_sl_n_auth_base, t_sl_n_auth_dc
    global c_bl_nn_auth_base, c_bl_nn_auth_dc, c_sl_nn_auth_base, c_sl_nn_auth_dc, t_bl_nn_auth_base, t_bl_nn_auth_dc, t_sl_nn_auth_base, t_sl_nn_auth_dc
    global c_bl_n_auth_rec, c_bl_nn_auth_rec, c_sl_n_auth_rec, c_sl_nn_auth_rec, t_bl_n_auth_rec, t_bl_nn_auth_rec, t_sl_n_auth_rec, t_sl_nn_auth_rec
    global c_bl_n_auth_rec_long, c_bl_nn_auth_rec_long, c_sl_n_auth_rec_long, c_sl_nn_auth_rec_long, t_bl_n_auth_rec_long, t_bl_nn_auth_rec_long, t_sl_n_auth_rec_long, t_sl_nn_auth_rec_long
    global c_al_n_auth_base, c_al_n_auth_dc, c_al_n_auth_rec, c_al_n_auth_rec_long, c_al_nn_auth_base, c_al_nn_auth_dc, c_al_nn_auth_rec, c_al_nn_auth_rec_long, t_al_n_auth_base, t_al_n_auth_dc, t_al_n_auth_rec, t_al_n_auth_rec_long, t_al_nn_auth_base, t_al_nn_auth_dc, t_al_nn_auth_rec, t_al_nn_auth_rec_long


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

    if with_app_links and no_score_links:
        for k in graph.keys():
            graph[k] = List.empty_list(int64)

    size = max(graph, key=int) +1 # +1 to account for starting at 0 index
    M = np.zeros((size, size), dtype=bool)
    to_adj_matrix(graph, M)

    # add APP links
    if with_app_links:
        chats_last_node = np.zeros(len(chats))
        for n in range(M.shape[0]):
            last_node_of_chat = int(chats_last_node[node_chat[n]])
            # if last_node_of_chat != 0:
            M[n, last_node_of_chat] = True
            chats_last_node[node_chat[n]] = n
        

    try:
        # print("Normalized")
        auth_base_n, auth_dc_n, auth_base_nn, auth_dc_nn, auth_rec_n, auth_rec_nn, auth_rec_long_n, auth_rec_long_nn = break_target(
            M.copy(), node_agent, agents, chats, node_chat, with_app_links, no_score_links )
        # print()
        # print("Not Normalized")
        # auth_base_nn, auth_dc_nn = break_target(
        # M.copy(), node_agent, agents, norm=False)

        if conf:
            if with_app_links:
                if no_score_links:
                    c_al_n_auth_base.put(auth_base_n)
                    c_al_n_auth_dc.put(auth_dc_n)
                    c_al_n_auth_rec.put(auth_rec_n)
                    c_al_n_auth_rec_long.put(auth_rec_long_n)
                    c_al_nn_auth_base.put(auth_base_nn)
                    c_al_nn_auth_dc.put(auth_dc_nn)
                    c_al_nn_auth_rec.put(auth_rec_nn)
                    c_al_nn_auth_rec_long.put(auth_rec_long_nn)
                else:
                    c_bl_n_auth_base.put(auth_base_n)
                    c_bl_n_auth_dc.put(auth_dc_n)
                    c_bl_n_auth_rec.put(auth_rec_n)
                    c_bl_n_auth_rec_long.put(auth_rec_long_n)
                    c_bl_nn_auth_base.put(auth_base_nn)
                    c_bl_nn_auth_dc.put(auth_dc_nn)
                    c_bl_nn_auth_rec.put(auth_rec_nn)
                    c_bl_nn_auth_rec_long.put(auth_rec_long_nn)
            else:
                c_sl_n_auth_base.put(auth_base_n)
                c_sl_n_auth_dc.put(auth_dc_n)
                c_sl_n_auth_rec.put(auth_rec_n)
                c_sl_n_auth_rec_long.put(auth_rec_long_n)
                c_sl_nn_auth_base.put(auth_base_nn)
                c_sl_nn_auth_dc.put(auth_dc_nn)
                c_sl_nn_auth_rec.put(auth_rec_nn)
                c_sl_nn_auth_rec_long.put(auth_rec_long_nn)
        else:
            if with_app_links:
                if no_score_links:
                    t_al_n_auth_base.put(auth_base_n)
                    t_al_n_auth_dc.put(auth_dc_n)
                    t_al_n_auth_rec.put(auth_rec_n)
                    t_al_n_auth_rec_long.put(auth_rec_long_n)
                    t_al_nn_auth_base.put(auth_base_nn)
                    t_al_nn_auth_dc.put(auth_dc_nn)
                    t_al_nn_auth_rec.put(auth_rec_nn)
                    t_al_nn_auth_rec_long.put(auth_rec_long_nn)
                else:
                    t_bl_n_auth_base.put(auth_base_n)
                    t_bl_n_auth_dc.put(auth_dc_n)
                    t_bl_n_auth_rec.put(auth_rec_n)
                    t_bl_n_auth_rec_long.put(auth_rec_long_n)
                    t_bl_nn_auth_base.put(auth_base_nn)
                    t_bl_nn_auth_dc.put(auth_dc_nn)
                    t_bl_nn_auth_rec.put(auth_rec_nn)
                    t_bl_nn_auth_rec_long.put(auth_rec_long_nn)
            else:
                t_sl_n_auth_base.put(auth_base_n)
                t_sl_n_auth_dc.put(auth_dc_n)
                t_sl_n_auth_rec.put(auth_rec_n)
                t_sl_n_auth_rec_long.put(auth_rec_long_n)
                t_sl_nn_auth_base.put(auth_base_nn)
                t_sl_nn_auth_dc.put(auth_dc_nn)
                t_sl_nn_auth_rec.put(auth_rec_nn)
                t_sl_nn_auth_rec_long.put(auth_rec_long_nn)
    except Exception as err:
        print("Exception occured:", err)
    # print()
    # return M


def break_target(A, node_agent, agents, chats, node_chat, with_app_links, no_score_links):
    # print(A.shape)
    k = 1
    M = A.copy()
    G_n = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    G_t = nx.transitive_closure(G_n)
    M_t = nx.to_numpy_array(G_t, dtype=np.float64)
    auth_base_nn = _hits.hits_authorities(M_t, normalized=False)
    while True:
        G_n = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
        stats = nx.betweenness_centrality(G_n)
        node = max(stats, key=lambda key: stats[key])
        M = np.delete(M, node, axis=0)
        M = np.delete(M, node, axis=1)
        G_n.remove_node(node)
        if not nx.is_weakly_connected(G_n):
            auth_dc_nn = np.zeros(len(G_n.nodes()), dtype=np.float64)
            auth_dc_index = 0
            subgraphs = [G_n.subgraph(
                c).copy() for c in nx.connected_components(G_n.to_undirected())]
            for G_s in subgraphs:
                if len(G_s.nodes()) == 1:
                    auth_dc_index += 1
                    continue
                G_ts = nx.transitive_closure(G_s)
                M_ts = nx.to_numpy_array(G_ts, dtype=np.float64)
                next_index = auth_dc_index + len(G_s.nodes())
                auth_dc_nn[auth_dc_index:next_index] = _hits.hits_authorities(
                    M_ts, normalized=False)
                auth_dc_index = next_index
            # normalizing
            auth_base_n = auth_base_nn / auth_base_nn.sum()
            auth_dc_n = auth_dc_nn / auth_dc_nn.sum()
            # recovery
            auth_rec_n, auth_rec_nn, auth_rec_long_n, auth_rec_long_nn = recover(chats, agents, node_chat, node_agent, G_n, with_app_links, no_score_links)
            return auth_base_n, auth_dc_n, auth_base_nn, auth_dc_nn, auth_rec_n, auth_rec_nn, auth_rec_long_n, auth_rec_long_nn
        k += 1

def recover(chats, agents, node_chat, node_agent, G, with_app_links, no_score_links):
    for chat in chats:
        chat.current_nodes = List.empty_list(int64)
        for n in chat.current_nodes :
            if n in G.nodes():
                chat.current_nodes.append(n)


    nodes = list(G.nodes())
    index = max(nodes)+1
    G_old = G.copy()
    # print("Dummies:",[n for n in range(max(G.nodes())) if not n in G.nodes()])
    G.add_nodes_from([n for n in range(max(G.nodes())) if not n in G.nodes()])

    # print(nx.is_connected(G.to_undirected()), index, [n for n in range(max(G.nodes())) if not n in G.nodes()])
    for agent_id in range(len(agents)):
        new_node = index+agent_id
        G.add_node(new_node) 
        for node in nodes:
            if G_old.in_degree(node) == 0 and node_chat[node] in agents[agent_id].chats:
                G.add_edge(new_node, node)
                node_chat[new_node] = -1
                node_agent[new_node] = agent_id
                chats[node_chat[node]].current_nodes.append(new_node)
        # print(index, agent_id, new_node, G.edges(new_node))

    # print(nx.is_connected(G.to_undirected()), index, [n for n in range(max(G.nodes())) if not n in G.nodes()])
    G_t = nx.transitive_closure(G)
    M = nx.to_numpy_array(G, dtype=np.float64)
    M_t = nx.to_numpy_array(G_t, dtype=np.float64)

    auth_rec_nn =_hits.hits_authorities(M_t, normalized=False)
    auth_rec_n = auth_rec_nn / auth_rec_nn.sum()

    rec_graph = from_adj_matrix(G)
    rec_graph_transitive = from_adj_matrix(G_t)
    rec_game = generate(chats,agents,rounds=10)
    
    run(rec_game, chats, agents, rec_graph, rec_graph_transitive, node_chat, node_agent, rounds=10)
    if with_app_links and no_score_links:
        for k in rec_game.keys():
            rec_graph[k] = List.empty_list(int64)

    
    size = max(rec_graph, key=int) + 1 # +1 to account for starting at 0 index
    M = np.zeros((size,size), dtype=np.float64)
    to_adj_matrix(rec_graph, M)

    # add APP links
    if with_app_links:
        chats_last_node = np.zeros(len(chats))
        for n in rec_graph.keys(): #range(M.shape[0]):
            last_node_of_chat = int(chats_last_node[node_chat[n]])
            if last_node_of_chat <= 0:
                M[n, last_node_of_chat] = True
            chats_last_node[node_chat[n]] = n

    G_n = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    G_t = nx.transitive_closure(G_n)
    M_t = nx.to_numpy_array(G_t, dtype=np.float64)

    auth_rec_long_nn =_hits.hits_authorities(M_t, normalized=False)
    auth_rec_long_n = auth_rec_long_nn / auth_rec_long_nn.sum()
    # auth_rec_long_nn = []
    # auth_rec_long_n = []

    return auth_rec_n, auth_rec_nn, auth_rec_long_n, auth_rec_long_nn


def sim(its):
    # # generate agent graph for CHATS OF TWOS
    # beta model
    G_a = nx.generators.random_graphs.connected_watts_strogatz_graph(
        NUMBER_OF_AGENTS, 4, 0.25)
    # # random model
    # while True:
    #     G_a = nx.gnp_random_graph(NUMBER_OF_AGENTS,0.4)
    #     if nx.is_connected(G_a):
    #         break
    # # fill in agents_in_chats
    agents_in_chats = []
    # fill in from graph for CHATS OF TWOS
    agents_in_chats = list(map(list, list(G_a.edges())))
    # # fill in sample random for CHATS OF VARIOUS # timeline style
    for n in G_a.nodes():
        agents_in_chats.append(list(G_a.neighbors(n)))

    a = Agent(List.empty_list(int64))
    agents = List.empty_list(typeof(a))

    c = Chat(List.empty_list(int64), List([0]))
    chats = List.empty_list(typeof(c))

    chats, agents = fill_chats_and_agents(chats, agents, agents_in_chats)

    ##############
    #    Game    #
    ##############
    game = generate(chats, agents)

    # confidential
    # print("CONFIDENTIAL")
    # print() app
    run_game(game, chats, agents, with_app_links=True, conf=True, no_score_links=True)
    # print() both
    run_game(game, chats, agents, with_app_links=True, conf=True)
    # print() score
    run_game(game, chats, agents, with_app_links=False, conf=True)
    # print()

    # print()
    # defines read rights to chat: full transparency
    for agent in agents:
        agent.chats = List(range(len(chats)))

    # transparent
    # print("TRANSPARENT")
    # print() app
    run_game(game, chats, agents, with_app_links=True, conf=False, no_score_links=True)
    # print() both
    run_game(game, chats, agents, with_app_links=True, conf=False)
    # print() score
    run_game(game, chats, agents, with_app_links=False, conf=False)
    # print()
    print(f"{its+1} done.")


def get_box_plot_data(labels, bp):
    rows_list = []
    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)

def add_plot(data, title):
    fig, ax = plt.subplots()
    if title[-2:] == '_n':
        ax.set_ylim(-0.00025,  0.0105) # 0.0235) #
        ax.set_yticks([0, 0.001, 0.002, 0.003, 0.004,
                       0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 
                    #    0.011, 0.012, 0.013, 0.014,
                    #    0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023 
                       ])
    ax.set_title(title)
    labels = ['base', 'disconnected', 'reconnect' , 'recovery'] # ] #
    bp = ax.boxplot(data, labels=labels, showfliers=False)
    plt.savefig(f'./img/{title}.png')
    # print(np.median(data[0]), np.median(data[1]))
    print()
    print(title)
    plot_data = get_box_plot_data(labels, bp)
    plot_data.to_pickle(f'./data/{title}.pkl')
    print(plot_data)
    print()


def init(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, r0, r1, r2, r3, r4, r5, r6, r7, r8,r9,r10,r11,r12,r13,r14,r15,z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15):
    global c_bl_n_auth_base, c_bl_n_auth_dc, c_sl_n_auth_base, c_sl_n_auth_dc, t_bl_n_auth_base, t_bl_n_auth_dc, t_sl_n_auth_base, t_sl_n_auth_dc
    global c_bl_nn_auth_base, c_bl_nn_auth_dc, c_sl_nn_auth_base, c_sl_nn_auth_dc, t_bl_nn_auth_base, t_bl_nn_auth_dc, t_sl_nn_auth_base, t_sl_nn_auth_dc
    global c_bl_n_auth_rec, c_bl_nn_auth_rec, c_sl_n_auth_rec, c_sl_nn_auth_rec, t_bl_n_auth_rec, t_bl_nn_auth_rec, t_sl_n_auth_rec, t_sl_nn_auth_rec
    global c_bl_n_auth_rec_long, c_bl_nn_auth_rec_long, c_sl_n_auth_rec_long, c_sl_nn_auth_rec_long, t_bl_n_auth_rec_long, t_bl_nn_auth_rec_long, t_sl_n_auth_rec_long, t_sl_nn_auth_rec_long
    global c_al_n_auth_base, c_al_n_auth_dc, c_al_n_auth_rec, c_al_n_auth_rec_long, c_al_nn_auth_base, c_al_nn_auth_dc, c_al_nn_auth_rec, c_al_nn_auth_rec_long, t_al_n_auth_base, t_al_n_auth_dc, t_al_n_auth_rec, t_al_n_auth_rec_long, t_al_nn_auth_base, t_al_nn_auth_dc, t_al_nn_auth_rec, t_al_nn_auth_rec_long



    c_bl_n_auth_base = a0
    c_bl_n_auth_dc = a1
    c_bl_nn_auth_base = a2
    c_bl_nn_auth_dc = a3
    c_sl_n_auth_base = a4
    c_sl_n_auth_dc = a5
    c_sl_nn_auth_base = a6
    c_sl_nn_auth_dc = a7
    t_bl_n_auth_base = a8
    t_bl_n_auth_dc = a9
    t_bl_nn_auth_base = a10
    t_bl_nn_auth_dc = a11
    t_sl_n_auth_base = a12
    t_sl_n_auth_dc = a13
    t_sl_nn_auth_base = a14
    t_sl_nn_auth_dc = a15
    c_bl_n_auth_rec = r0
    c_bl_nn_auth_rec = r1
    c_sl_n_auth_rec = r2
    c_sl_nn_auth_rec = r3
    t_bl_n_auth_rec = r4
    t_bl_nn_auth_rec = r5
    t_sl_n_auth_rec = r6
    t_sl_nn_auth_rec = r7
    c_bl_n_auth_rec_long = r8
    c_bl_nn_auth_rec_long = r9
    c_sl_n_auth_rec_long = r10
    c_sl_nn_auth_rec_long = r11
    t_bl_n_auth_rec_long = r12
    t_bl_nn_auth_rec_long = r13
    t_sl_n_auth_rec_long = r14
    t_sl_nn_auth_rec_long = r15
    c_al_n_auth_base = z0
    c_al_n_auth_dc = z1
    c_al_n_auth_rec = z2
    c_al_n_auth_rec_long = z3
    c_al_nn_auth_base = z4
    c_al_nn_auth_dc = z5
    c_al_nn_auth_rec = z6
    c_al_nn_auth_rec_long = z7
    t_al_n_auth_base = z8
    t_al_n_auth_dc = z9
    t_al_n_auth_rec = z10
    t_al_n_auth_rec_long = z11
    t_al_nn_auth_base = z12
    t_al_nn_auth_dc = z13
    t_al_nn_auth_rec = z14
    t_al_nn_auth_rec_long = z15



def extract(queue):
    tmp = []
    while not queue.empty():
        tmp.extend(queue.get())
    return tmp


if __name__ == "__main__":
    # c confidential
    # bl both links
    # n normalized
    c_bl_n_auth_base = mp.Queue()
    c_bl_n_auth_dc = mp.Queue()
    c_bl_nn_auth_base = mp.Queue()
    c_bl_nn_auth_dc = mp.Queue()
    c_sl_n_auth_base = mp.Queue()
    c_sl_n_auth_dc = mp.Queue()
    c_sl_nn_auth_base = mp.Queue()
    c_sl_nn_auth_dc = mp.Queue()
    t_bl_n_auth_base = mp.Queue()
    t_bl_n_auth_dc = mp.Queue()
    t_bl_nn_auth_base = mp.Queue()
    t_bl_nn_auth_dc = mp.Queue()
    t_sl_n_auth_base = mp.Queue()
    t_sl_n_auth_dc = mp.Queue()
    t_sl_nn_auth_base = mp.Queue()
    t_sl_nn_auth_dc = mp.Queue()
    c_bl_n_auth_rec = mp.Queue()
    c_bl_nn_auth_rec = mp.Queue()
    c_sl_n_auth_rec = mp.Queue()
    c_sl_nn_auth_rec = mp.Queue()
    t_bl_n_auth_rec = mp.Queue()
    t_bl_nn_auth_rec = mp.Queue()
    t_sl_n_auth_rec = mp.Queue()
    t_sl_nn_auth_rec = mp.Queue()
    c_bl_n_auth_rec_long = mp.Queue()
    c_bl_nn_auth_rec_long = mp.Queue()
    c_sl_n_auth_rec_long = mp.Queue()
    c_sl_nn_auth_rec_long = mp.Queue()
    t_bl_n_auth_rec_long = mp.Queue()
    t_bl_nn_auth_rec_long = mp.Queue()
    t_sl_n_auth_rec_long = mp.Queue()
    t_sl_nn_auth_rec_long = mp.Queue()
    c_al_n_auth_base = mp.Queue()
    c_al_n_auth_dc = mp.Queue()
    c_al_n_auth_rec = mp.Queue()
    c_al_n_auth_rec_long = mp.Queue()
    c_al_nn_auth_base = mp.Queue()
    c_al_nn_auth_dc = mp.Queue()
    c_al_nn_auth_rec = mp.Queue()
    c_al_nn_auth_rec_long = mp.Queue()
    t_al_n_auth_base = mp.Queue()
    t_al_n_auth_dc = mp.Queue()
    t_al_n_auth_rec = mp.Queue()
    t_al_n_auth_rec_long = mp.Queue()
    t_al_nn_auth_base = mp.Queue()
    t_al_nn_auth_dc = mp.Queue()
    t_al_nn_auth_rec = mp.Queue()
    t_al_nn_auth_rec_long = mp.Queue()


    iteration_ids = range(0, ITERATIONS)
    pool = mp.Pool(processes=CORES, initializer=init, initargs=(
        c_bl_n_auth_base, c_bl_n_auth_dc, c_bl_nn_auth_base, c_bl_nn_auth_dc, c_sl_n_auth_base, c_sl_n_auth_dc, c_sl_nn_auth_base, c_sl_nn_auth_dc, t_bl_n_auth_base, t_bl_n_auth_dc, t_bl_nn_auth_base, t_bl_nn_auth_dc, t_sl_n_auth_base, t_sl_n_auth_dc, t_sl_nn_auth_base, t_sl_nn_auth_dc, c_bl_n_auth_rec, c_bl_nn_auth_rec, c_sl_n_auth_rec, c_sl_nn_auth_rec, t_bl_n_auth_rec, t_bl_nn_auth_rec, t_sl_n_auth_rec, t_sl_nn_auth_rec,  c_bl_n_auth_rec_long, c_bl_nn_auth_rec_long, c_sl_n_auth_rec_long, c_sl_nn_auth_rec_long, t_bl_n_auth_rec_long, t_bl_nn_auth_rec_long, t_sl_n_auth_rec_long, t_sl_nn_auth_rec_long,  c_al_n_auth_base,  c_al_n_auth_dc,  c_al_n_auth_rec,  c_al_n_auth_rec_long,  c_al_nn_auth_base,  c_al_nn_auth_dc,  c_al_nn_auth_rec,  c_al_nn_auth_rec_long,  t_al_n_auth_base,  t_al_n_auth_dc,  t_al_n_auth_rec,  t_al_n_auth_rec_long,  t_al_nn_auth_base,  t_al_nn_auth_dc,  t_al_nn_auth_rec,  t_al_nn_auth_rec_long, )
    )
    results = pool.map(sim, iteration_ids)
    pool.terminate()
    pool.join()
    print(f"Pool done with all {ITERATIONS} iterations!")

    c_bl_n_auth_base = extract(c_bl_n_auth_base)
    c_bl_n_auth_dc = extract(c_bl_n_auth_dc)
    c_bl_nn_auth_base = extract(c_bl_nn_auth_base)
    c_bl_nn_auth_dc = extract(c_bl_nn_auth_dc)
    c_sl_n_auth_base = extract(c_sl_n_auth_base)
    c_sl_n_auth_dc = extract(c_sl_n_auth_dc)
    c_sl_nn_auth_base = extract(c_sl_nn_auth_base)
    c_sl_nn_auth_dc = extract(c_sl_nn_auth_dc)
    t_bl_n_auth_base = extract(t_bl_n_auth_base)
    t_bl_n_auth_dc = extract(t_bl_n_auth_dc)
    t_bl_nn_auth_base = extract(t_bl_nn_auth_base)
    t_bl_nn_auth_dc = extract(t_bl_nn_auth_dc)
    t_sl_n_auth_base = extract(t_sl_n_auth_base)
    t_sl_n_auth_dc = extract(t_sl_n_auth_dc)
    t_sl_nn_auth_base = extract(t_sl_nn_auth_base)
    t_sl_nn_auth_dc = extract(t_sl_nn_auth_dc)
    c_bl_n_auth_rec = extract(c_bl_n_auth_rec)
    c_bl_nn_auth_rec = extract(c_bl_nn_auth_rec)
    c_sl_n_auth_rec = extract(c_sl_n_auth_rec)
    c_sl_nn_auth_rec = extract(c_sl_nn_auth_rec)
    t_bl_n_auth_rec = extract(t_bl_n_auth_rec)
    t_bl_nn_auth_rec = extract(t_bl_nn_auth_rec)
    t_sl_n_auth_rec = extract(t_sl_n_auth_rec)
    t_sl_nn_auth_rec = extract(t_sl_nn_auth_rec)
    c_bl_n_auth_rec_long = extract(c_bl_n_auth_rec_long)
    c_bl_nn_auth_rec_long = extract(c_bl_nn_auth_rec_long)
    c_sl_n_auth_rec_long = extract(c_sl_n_auth_rec_long)
    c_sl_nn_auth_rec_long = extract(c_sl_nn_auth_rec_long)
    t_bl_n_auth_rec_long = extract(t_bl_n_auth_rec_long)
    t_bl_nn_auth_rec_long = extract(t_bl_nn_auth_rec_long)
    t_sl_n_auth_rec_long = extract(t_sl_n_auth_rec_long)
    t_sl_nn_auth_rec_long = extract(t_sl_nn_auth_rec_long)
    c_al_n_auth_base = extract(c_al_n_auth_base )
    c_al_n_auth_dc = extract(c_al_n_auth_dc )
    c_al_n_auth_rec = extract(c_al_n_auth_rec )
    c_al_n_auth_rec_long = extract(c_al_n_auth_rec_long)
    c_al_nn_auth_base = extract(c_al_nn_auth_base )
    c_al_nn_auth_dc = extract(c_al_nn_auth_dc )
    c_al_nn_auth_rec = extract(c_al_nn_auth_rec )
    c_al_nn_auth_rec_long = extract(c_al_nn_auth_rec_long)
    t_al_n_auth_base = extract(t_al_n_auth_base )
    t_al_n_auth_dc = extract(t_al_n_auth_dc )
    t_al_n_auth_rec = extract(t_al_n_auth_rec )
    t_al_n_auth_rec_long = extract(t_al_n_auth_rec_long)
    t_al_nn_auth_base = extract(t_al_nn_auth_base )
    t_al_nn_auth_dc = extract(t_al_nn_auth_dc )
    t_al_nn_auth_rec = extract(t_al_nn_auth_rec )
    t_al_nn_auth_rec_long = extract(t_al_nn_auth_rec_long)

    # print("c_bl_n")
    add_plot([c_bl_n_auth_base, c_bl_n_auth_dc, c_bl_n_auth_rec, c_bl_n_auth_rec_long], "c_bl_n") #  ], "c_bl_n")#
    # print()
    # print("c_bl_nn")
    add_plot([c_bl_nn_auth_base, c_bl_nn_auth_dc, c_bl_nn_auth_rec, c_bl_nn_auth_rec_long], "c_bl_nn") #  ], "c_bl_nn")#
    # print()
    # print("c_sl_n")
    add_plot([c_sl_n_auth_base, c_sl_n_auth_dc, c_sl_n_auth_rec, c_sl_n_auth_rec_long], "c_sl_n") #  ], "c_sl_n")#
    # print()
    # print("c_sl_nn")
    add_plot([c_sl_nn_auth_base, c_sl_nn_auth_dc, c_sl_nn_auth_rec, c_sl_nn_auth_rec_long], "c_sl_nn") #  ], "c_sl_nn")#
    # print()
    # print("c_al_n")
    add_plot([c_al_n_auth_base, c_al_n_auth_dc, c_al_n_auth_rec, c_al_n_auth_rec_long], "c_al_n") #  ], "c_al_n")#
    # print()
    # print("c_al_nn")
    add_plot([c_al_nn_auth_base, c_al_nn_auth_dc, c_al_nn_auth_rec, c_al_nn_auth_rec_long], "c_al_nn") #  ], "c_al_nn")#
    # print()
    # print("t_bl_n")
    add_plot([t_bl_n_auth_base, t_bl_n_auth_dc, t_bl_n_auth_rec, t_bl_n_auth_rec_long], "t_bl_n") #  ], "t_bl_n")#
    # print()
    # print("t_bl_nn")
    add_plot([t_bl_nn_auth_base, t_bl_nn_auth_dc, t_bl_nn_auth_rec, t_bl_nn_auth_rec_long], "t_bl_nn") #  ], "t_bl_nn")#
    # print()
    # print("t_sl_n")
    add_plot([t_sl_n_auth_base, t_sl_n_auth_dc, t_sl_n_auth_rec, t_sl_n_auth_rec_long], "t_sl_n") #  ], "t_sl_n")#
    # print()
    # print("t_sl_nn")
    add_plot([t_sl_nn_auth_base, t_sl_nn_auth_dc, t_sl_nn_auth_rec, t_sl_nn_auth_rec_long], "t_sl_nn") #  ], "t_sl_nn")#
    # print()
    # print("t_al_n")
    add_plot([t_al_n_auth_base, t_al_n_auth_dc, t_al_n_auth_rec, t_al_n_auth_rec_long], "t_al_n") #  ], "t_al_n")#
    # print()
    # print("t_al_nn")
    add_plot([t_al_nn_auth_base, t_al_nn_auth_dc, t_al_nn_auth_rec, t_al_nn_auth_rec_long], "t_al_nn") #  ], "t_al_nn")#
    # print()

    # plt.show()

    # print()
    # print((c_bl_n_auth_base), "\n", (c_bl_n_auth_dc))

    # print()
    # print((c_bl_nn_auth_base), "\n", (c_bl_nn_auth_dc))

    # print()
    # # print((c_sl_n_auth_base), "\n",(c_sl_n_auth_dc))

    # print()
    # print((c_sl_nn_auth_base), "\n", (c_sl_nn_auth_dc))

    # print()
    # # print((t_bl_n_auth_base), "\n",(t_bl_n_auth_dc))

    # print()
    # print((t_bl_nn_auth_base), "\n", (t_bl_nn_auth_dc))

    # print()
    # # print((t_sl_n_auth_base), "\n",(t_sl_n_auth_dc))

    # print()
    # print((t_sl_nn_auth_base), "\n", (t_sl_nn_auth_dc))
