import streamlit as st
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit.components.v1 as components 
from pyvis.network import Network
from operator import itemgetter 
from collections import Counter
from networkx.drawing.nx_agraph import graphviz_layout 
import community.community_louvain as community2
from networkx.algorithms import community

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.wallpapersafari.com/11/31/stKmlF.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
add_bg_from_url() 

image_url = "https://pngimg.com/uploads/star_wars_logo/star_wars_logo_PNG20.png"
st.sidebar.image(image_url, use_column_width=True)

st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")

uploaded_file = st.sidebar.file_uploader("Carica il file JSON:", type="json")

if uploaded_file is None:

    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    
    st.sidebar.image("https://studcar.ru/wp-content/uploads/2020/10/logo-lumsa-1.png", use_column_width=True)
    
    st.markdown("<h1 style='text-align: center;'>WELCOME YOU ARE</h1>", unsafe_allow_html=True)
    
    image_url = "https://1.bp.blogspot.com/-R9novqwzprA/V1LnhAeWayI/AAAAAAAAEuQ/6jOkfjgu6KsauBmAMBGBuWZmRHpkoa8QACLcB/s1600/Yodar.png"

    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img src="{image_url}" width="300" /></div>',
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center;'>INSERT THE DATASET YOU MUST</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    if uploaded_file.name not in ["starwars-full-interactions-allCharacters.json", "starwars-episode-1-interactions-allCharacters.json", "starwars-episode-2-interactions-allCharacters.json", "starwars-episode-3-interactions-allCharacters.json", "starwars-episode-4-interactions-allCharacters.json", "starwars-episode-5-interactions-allCharacters.json", "starwars-episode-6-interactions-allCharacters.json", "starwars-episode-7-interactions-allCharacters.json"]:
        st.sidebar.write(" ")
        st.sidebar.write(" ")
        st.sidebar.write(" ")

        st.sidebar.image("https://studcar.ru/wp-content/uploads/2020/10/logo-lumsa-1.png", use_column_width=True)

        st.markdown("<h1 style='text-align: center;'>THE FILE IS WRONG</h1>", unsafe_allow_html=True)

        image_url = "https://www.pngall.com/wp-content/uploads/9/Star-Wars-Obi-Wan-Kenobi-PNG-File.png"

        st.markdown(
            f'<div style="display: flex; justify-content: center;"><img src="{image_url}" width="400" /></div>',
            unsafe_allow_html=True
        )

        st.markdown("<h1 style='text-align: center;'>YOU ARE NOT THE CHOSEN ONE!</h1>", unsafe_allow_html=True)
    else:
        isolate=st.sidebar.checkbox("Rimuovere nodo isolato")
        dynamic=st.sidebar.checkbox('Grafico dinamico')
        
        #st.sidebar.write(" ")

        st.sidebar.image("https://studcar.ru/wp-content/uploads/2020/10/logo-lumsa-1.png", use_column_width=True)

        st.markdown("<h1 style='text-align: center; background-color: black; border: 4px solid yellow; color: white; padding: 10px;'>STAR WARS SOCIAL NETWORK ANALYSIS</h1>", unsafe_allow_html=True)
        
        st.title(" ")

        data = json.load(uploaded_file)
        nodes = data["nodes"]
        links = data["links"]

        G = nx.Graph()

        for node in nodes:
            G.add_node(node["name"], value=node["value"], colour=node["colour"])

        for link in links:
            source = nodes[link["source"]]["name"]
            target = nodes[link["target"]]["name"]
            value = link["value"]
            G.add_edge(source, target, value=value)

        if isolate:
            G.remove_nodes_from(list(nx.isolates(G)))

        if dynamic:
            net=Network(height="600pt", width="100%", bgcolor="#222222", font_color="white",select_menu=True,filter_menu=True,cdn_resources='remote',notebook=True)

            net.from_nx(G)


            net.show_buttons(filter_=["physics"])

            net.show('test.html')

            HtmlFile = open("test.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=1500, width=1000)
        else:
            layout_option = st.selectbox("Seleziona il layout:", ("Random", "Circular"))

            if layout_option == "Random":
                fig, ax = plt.subplots(figsize=(15, 15))
                pos = nx.spring_layout(G, seed=12, k=0.45)
                
                options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 0.95}
                nx.draw_networkx_nodes(G, pos, node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
                
                nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.25)
                
                nx.draw_networkx_labels(G, pos, font_size=5, alpha=0.7)
                st.pyplot(plt)
            elif layout_option == "Circular":
                fig, ax = plt.subplots(figsize=(15, 15))
                pos = nx.circular_layout(G)
                
                options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 0.95}
                nx.draw_networkx_nodes(G, pos, node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
                
                nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.25)
                
                nx.draw_networkx_labels(G, pos, font_size=5, alpha=0.7)
                st.pyplot(plt)
            
            st.subheader("Informazioni sul grafo")
            first_column, second_column = st.columns((2, 2))
            with first_column:
                st.write("Numero di nodi:", f'<span style="color: yellow;">{G.number_of_nodes()}</span>', unsafe_allow_html=True)
                st.write("Numero di collegamenti:", f'<span style="color: yellow;">{G.number_of_edges()}</span>', unsafe_allow_html=True)
                st.write("Grado medio:", f'<span style="color: yellow;">{round(np.mean([d for _, d in G.degree()]), 3)}</span>', unsafe_allow_html=True)
                st.write("Densità:", f'<span style="color: yellow;">{round(nx.density(G), 3)}</span>', unsafe_allow_html=True)
                if isolate:
                    st.write("Lunghezza media del percorso più breve", f'<span style="color: yellow;">{round(nx.average_shortest_path_length(G), 3)}</span>', unsafe_allow_html=True)
                #st.write("Percorso minimo medio:", f'<span style="color: yellow;">{round(np.mean([np.mean(list(spl.values())) for spl in dict(nx.all_pairs_shortest_path_length(G)).values()]), 2)}</span>', unsafe_allow_html=True)
            with second_column:
                st.write("Numero di triangoli:", f'<span style="color: yellow;">{round(sum(list(nx.triangles(G).values())), 2)/3}</span>', unsafe_allow_html=True)
                st.write("Coefficiente medio di clustering:", f'<span style="color: yellow;">{round(nx.average_clustering(G), 3)}</span>', unsafe_allow_html=True)
                st.write("Coefficiente globale di clustering:", f'<span style="color: yellow;">{round(nx.transitivity(G), 3)}</span>', unsafe_allow_html=True)
                st.write("Coefficiente di assortatività:", f'<span style="color: yellow;">{round(nx.degree_assortativity_coefficient(G), 3)}</span>', unsafe_allow_html=True)
                if isolate:
                    st.write("Diametro:", f'<span style="color: yellow;">{round(nx.diameter(G), 3)}</span>', unsafe_allow_html=True)
                    
            
            st.write(" ")
            
            matrix = st.checkbox("Mostra matrice di adiacenza")
            
            if matrix:
                
                adj_matrix = nx.to_numpy_matrix(G)

                fig, ax = plt.subplots(figsize=(20, 20))
                sns.heatmap(adj_matrix, cmap="Blues", annot=True, fmt="g", cbar=False,
                            xticklabels=G.nodes(), yticklabels=G.nodes())
                ax.set_title("Matrice di Adiacenza")
                st.pyplot(plt)
                
            matrix2 = st.checkbox("Mostra matrice di distanza")

            if matrix2:
                
                dist_matrix = np.array(nx.floyd_warshall_numpy(G))

                fig, ax = plt.subplots(figsize=(20, 20))
                sns.heatmap(dist_matrix, cmap="Blues", annot=True, fmt="g", cbar=False,
                            xticklabels=G.nodes(), yticklabels=G.nodes())
                ax.set_title("Matrice di Distanza")
                st.pyplot(plt.gcf())
            
            ######################

            st.title("Degree Centrality")
            fig, ax = plt.subplots(figsize=(15, 15))
            pos = nx.spring_layout(G, seed=12,k=0.85)

            options = {"edgecolors": "tab:gray", "alpha": 0.95}
            nx.draw_networkx_nodes(G, pos,node_size=[v * 3000 for v in nx.centrality.degree_centrality(G).values()],node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
            
            nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.25)
            
            nx.draw_networkx_labels(G,pos, font_size=5,alpha=0.8)
            st.pyplot(plt)

            first_column,second_column=st.columns((2,2))
            with first_column:
                st.subheader("Degree Centrality Score")
                for node, centrality in sorted(nx.centrality.degree_centrality(G).items(), key=lambda item: item[1], reverse=True)[:8]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            with second_column:
                st.subheader("Degree")
                for node, degree in sorted(G.degree, key=lambda item: item[1], reverse=True)[:8]:
                    st.write(node + ":", f'<span style="color: yellow;">{degree}</span>', unsafe_allow_html=True)
            
            st.subheader("Degree Distribution")
            
            graph_option = st.selectbox("Seleziona il tipo di grafico:", ("Scatter Plot", "Istogramma"))
            
            if graph_option == "Scatter Plot":
                #st.title("Degree Distribution")
                hist = nx.degree_histogram(G)
                plt.figure(figsize=(5, 5))
                plt.plot(range(0, len(hist)), hist, ".")
                plt.title("Degree Distribution")
                plt.xlabel("Degree")
                plt.ylabel("#Nodes")
                plt.loglog()
                st.pyplot(plt)
            elif graph_option == "Istogramma":
                degree_sequence = [G.degree(n) for n in G.nodes()]
                plt.figure(figsize=(8, 6))
                counts, bins, patches = plt.hist(degree_sequence, bins=100)
                plt.title("Degree Distribution")
                plt.xlabel("Degree")
                plt.ylabel("#Nodes")
                st.pyplot(plt)
                
            ######################

            st.title("Betweenness Centrality")
            fig, ax = plt.subplots(figsize=(15, 15))
            pos = nx.spring_layout(G, seed=12,k=0.85)
            
            options = {"edgecolors": "tab:gray", "alpha": 0.95}
            nx.draw_networkx_nodes(G, pos,node_size=[v * 2500 for v in nx.centrality.betweenness_centrality(G).values()], node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
            
            nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.25)
            
            nx.draw_networkx_labels(G,pos, font_size=5,alpha=0.8)
            st.pyplot(plt)
            
            st.subheader("Betweenness Centrality Score")
            first_column,second_column=st.columns((2,2))
            with first_column:
                for node, centrality in sorted(nx.centrality.betweenness_centrality(G).items(), key=lambda item: item[1], reverse=True)[:4]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            with second_column:
                for node, centrality in sorted(nx.centrality.betweenness_centrality(G).items(), key=lambda item: item[1], reverse=True)[4:8]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)

            ######################
            
            st.title("Closeness Centrality")
            fig, ax = plt.subplots(figsize=(15, 15))
            pos = nx.spring_layout(G, seed=12,k=0.9)
            
            options = {"edgecolors": "tab:gray", "alpha": 0.95}
            nx.draw_networkx_nodes(G, pos,node_size=[v * 2500 for v in nx.centrality.closeness_centrality(G).values()], node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
            
            nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.25)
            
            nx.draw_networkx_labels(G,pos, font_size=5,alpha=0.8)
            st.pyplot(plt)
            
            st.subheader("Closeness Centrality Score")
            first_column,second_column=st.columns((2,2))
            with first_column:
                for node, centrality in sorted(nx.centrality.closeness_centrality(G).items(), key=lambda item: item[1], reverse=True)[:4]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            with second_column:
                for node, centrality in sorted(nx.centrality.closeness_centrality(G).items(), key=lambda item: item[1], reverse=True)[4:8]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            
            ######################
            
            st.title("PageRank Centrality")
            fig, ax = plt.subplots(figsize=(20, 20))
            pos = nx.spring_layout(G, seed=12,k=1)
            
            options = {"edgecolors": "tab:gray", "alpha": 0.95}
            nx.draw_networkx_nodes(G, pos,node_size= [v * 20000 for v in nx.pagerank(G, alpha=0.9).values()], node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
            
            nx.draw_networkx_edges(G, pos, width=.7, alpha=0.25)
            
            nx.draw_networkx_labels(G,pos, font_size=5,alpha=0.9)
            st.pyplot(plt)
            
            st.subheader("PageRank Centrality Score")
            first_column,second_column=st.columns((2,2))
            with first_column:
                for node, centrality in sorted(nx.pagerank(G, alpha=0.9).items(), key=lambda x: x[1], reverse=True)[:4]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            with second_column:
                for node, centrality in sorted(nx.pagerank(G, alpha=0.9).items(), key=lambda x: x[1], reverse=True)[4:8]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
                    
            ######################
            
            st.title("Eigenvector Centrality")
            fig, ax = plt.subplots(figsize=(20, 20))
            pos = nx.spring_layout(G, seed=12,k=0.85)
            
            options = {"edgecolors": "tab:gray", "alpha": 0.95}
            nx.draw_networkx_nodes(G, pos,node_size= [v * 5000 for v in nx.centrality.eigenvector_centrality(G).values()], node_color=[node["colour"] for _, node in G.nodes(data=True)], **options)
            
            nx.draw_networkx_edges(G, pos, width=.7, alpha=0.25)
            
            nx.draw_networkx_labels(G,pos, font_size=5,alpha=0.9)
            st.pyplot(plt)
            
            st.subheader("Eigenvector Centrality Score")
            first_column,second_column=st.columns((2,2))
            with first_column:
                for node, centrality in sorted(nx.centrality.eigenvector_centrality(G).items(), key=lambda x: x[1], reverse=True)[:4]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            with second_column:
                for node, centrality in sorted(nx.centrality.eigenvector_centrality(G).items(), key=lambda x: x[1], reverse=True)[4:8]:
                    st.write(node + ":", f'<span style="color: yellow;">{round(centrality, 3)}</span>', unsafe_allow_html=True)
            
            ######################
            
            st.title("Bridges")
            plt.figure(figsize=(20, 20))
            nx.draw_networkx(G, pos=nx.spring_layout(G, k=0.9, seed=1), node_size=800, with_labels=True, width=0.35, alpha=0.8, node_color=[node["colour"] for _, node in G.nodes(data=True)])
            nx.draw_networkx_edges(G, pos=nx.spring_layout(G, k=0.9, seed=1), edgelist=list(nx.local_bridges(G, with_span=False)), width=1.75, edge_color="lawngreen")  # green color for local bridges
            nx.draw_networkx_edges(G, pos=nx.spring_layout(G, k=0.9, seed=1), edgelist=list(nx.bridges(G)), width=1.75, edge_color="r")  
            st.pyplot(plt)

            bridge_edges = list(nx.bridges(G))
            local_bridge_edges = list(nx.local_bridges(G, with_span=False))

            bridge_node_pairs = [(u, v) for u, v in bridge_edges]
            local_bridge_node_pairs = [(u, v) for u, v in local_bridge_edges]
            
            first_column,second_column=st.columns((2,2))
            with first_column:
                st.subheader("Coppie con bridge:")
                for pair in bridge_node_pairs:
                    node1, node2 = pair
                    st.write(f"{node1} <span style='color: yellow;'>-</span> {node2}", unsafe_allow_html=True)
            with second_column:
                st.subheader("Coppie con local bridge:")
                for pair in local_bridge_node_pairs:
                    node1, node2 = pair
                    st.write(f"{node1} <span style='color: yellow;'>-</span> {node2}", unsafe_allow_html=True)
            
            ######################
            
            st.title("Communities")
    
            community_option = st.selectbox("Seleziona il tipo di Community:", ("Louvain Algorithm", "Girvan-Newman Algorithm"))

            if community_option == "Louvain Algorithm":
                
                communities = community2.best_partition(G,random_state=42)

                fig, ax = plt.subplots(figsize=(15, 15))
                pos = nx.spring_layout(G, seed=12, k=0.45)
                node_colors = [communities[node] for node in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap="viridis")
                nx.draw_networkx_edges(G, pos, alpha=0.5)
                nx.draw_networkx_labels(G,pos, font_size=5,alpha=0.9)
                st.pyplot(plt)
                
                communitiez = st.checkbox("Vedi le communities")
                
                if communitiez:
                    community_results = {
                        'Personaggio': list(communities.keys()),
                        'Comunità': list(communities.values())
                    }

                    df = pd.DataFrame(community_results)

                    grouped_df = df.groupby('Comunità')

                    for group, data in grouped_df:
                        st.subheader(f"Community {group}:")
                        for character in data['Personaggio']:
                            st.write(character)
                
            elif community_option == "Girvan-Newman Algorithm":

                c = community.girvan_newman(G)

                node_groups = [list(i) for i in next(c)]

                node_indices = {node: idx for idx, node in enumerate(G.nodes())}

                colors = ['#3776ab', 'green', 'red', 'purple', 'orange', 'yellow', 'pink', 'cyan']

                plt.figure(figsize=(20, 20))
                pos = nx.spring_layout(G, seed=12, k=0.3)
                color_map = [0] * len(G)

                for i, group in enumerate(node_groups):
                    for node in group:
                        idx = node_indices[node]
                        color_map[idx] = colors[i % len(colors)]

                nx.draw_networkx_nodes(G, pos, node_size=400, node_color=color_map, **options)
                nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.25)
                nx.draw_networkx_labels(G, pos, font_size=5, alpha=0.9)

                st.pyplot(plt.gcf())
                
                communitiez2 = st.checkbox("Vedi le communities")
                
                if communitiez2:
                    for i, group in enumerate(node_groups):
                        st.subheader(f"Community {i}:")
                        for character in group:
                            st.write(character)
                
            ######################
                
            st.title("Ego Network")

            center_node = st.selectbox("Seleziona un personaggio", list(G.nodes()))

            radius = 1

            ego_graph = nx.ego_graph(G, center_node, radius=radius)

            node_colors = [node['colour'] for _, node in ego_graph.nodes(data=True)]

            pos = nx.spring_layout(ego_graph, seed=42)
            options = {
                'node_color': node_colors,
                'node_size': 200,
                'edge_color': 'gray',
                'width': 0.5,
                'with_labels': True,
                'font_size': 8,
                'font_color': 'black'
            }
            plt.figure(figsize=(8, 8))
            nx.draw_networkx(ego_graph, pos, **options)
            plt.title("Ego Network (Center: {})".format(center_node))
            st.pyplot(plt)

            ######################
            
            st.title("Shortest Path Lenght tra 2 Personaggi")

            character_1 = st.selectbox("Seleziona il primo personaggio", list(G.nodes()))
            character_2 = st.selectbox("Seleziona il secondo personaggio", list(G.nodes()))

            try:
                num_paths = nx.shortest_path_length(G, character_1, character_2)
            except nx.NetworkXNoPath:
                num_paths = 0

            st.write("La Shortest Path Length tra", character_1, "e", character_2, "è:", f'<span style="color: yellow;">{num_paths}</span>', unsafe_allow_html=True)









    
