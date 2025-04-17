import cv2
import os
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import networkx as nx
import matplotlib.pyplot as plt
from skan import Skeleton, summarize # 保持不变
import json
from similarity_index_of_label_graph_package import similarity_index_of_label_graph_class

def extract_fracture_network(image_path, threshold_value=128, visualize=False):
    """
    Extracts a topological network graph from a fracture mask image.

    Args:
        image_path (str): Path to the input mask image file.
        threshold_value (int): Binarization threshold (0-255). White areas (fractures) should be above this value.
        visualize (bool): Whether to display visualization results.

    Returns:
        networkx.Graph: The extracted fracture topology network graph.
                        Node attributes: 'pos' (coordinates (row, col))
                        Edge attributes: 'weight' (fracture segment length), 'path' (list of pixel coordinates forming the edge)
        np.ndarray: Original image (grayscale).
        np.ndarray: Extracted skeleton image (0 or 255).
    """
    # 1. Read image (grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")

    print(f"Image dimensions: {img.shape}")
    print(f"Image data type: {img.dtype}")
    #将img中非0的值变为255
    img[img != 0] = 255 
    # 2. Binarize (ensure 0 and 255)
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    binary_bool = binary_img > 0

    # 3. Skeletonize
    print("Extracting skeleton...")
    skeleton = skeletonize(binary_bool)
    skeleton_img = img_as_ubyte(skeleton)
    print("Skeleton extraction complete.")
    print(f"Number of white pixels in skeleton: {np.sum(skeleton)}")

    if np.sum(skeleton) == 0:
        print("Warning: Skeleton is empty. No fractures detected or fractures are too thin.")
        G = nx.Graph()
        if visualize:
            # --- Visualization for Empty Skeleton ---
            print("Generating visualization for empty skeleton...")
            plt.style.use('default') # Use default style
            plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ax = axes.ravel()

            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Original Mask Image') # English Title
            ax[0].axis('off')

            ax[1].imshow(skeleton_img, cmap='gray')
            ax[1].set_title('Extracted Skeleton (Empty)') # English Title
            ax[1].axis('off')

            plt.tight_layout()
            #plt.show()
            # --- End Visualization ---
        return G, img, skeleton_img

    # 4. Topological network extraction (using skan)
    print("Analyzing skeleton and extracting network...")
    skel_obj = Skeleton(skeleton)
    graph_df = summarize(skel_obj, separator='_')

    unique_node_ids_in_graph = np.unique(np.concatenate([graph_df['node_id_src'], graph_df['node_id_dst']]))
    num_nodes_in_graph = len(unique_node_ids_in_graph)

    print(f"Detected {num_nodes_in_graph} effective nodes (endpoints/junctions participating in edges)")
    print(f"Detected {len(graph_df)} edges (fracture segments)")

    # 5. Build NetworkX graph
    print("Building NetworkX graph...")
    G = nx.Graph() # Create empty graph

    node_coords = skel_obj.coordinates
    num_skeleton_pixels = len(node_coords)
    print(f"Skeleton coordinates array size (node_coords): {num_skeleton_pixels}")

    unique_node_ids_float = np.unique(np.concatenate([graph_df['node_id_src'], graph_df['node_id_dst']]))
    node_positions = {} # For plotting (col, row)

    print("Adding nodes to the graph...")
    for node_id_float in unique_node_ids_float:
        try:
            node_index = int(node_id_float)
            if 0 <= node_index < num_skeleton_pixels:
                r, c = node_coords[node_index]
                python_r = int(r)
                python_c = int(c)
                pos_tuple = (python_r, python_c)
                G.add_node(node_index, pos=pos_tuple)
                node_positions[node_index] = (python_c, python_r)
            else:
                print(f"Warning: Node ID {node_id_float} (index {node_index}) is out of bounds for coordinates array [0, {num_skeleton_pixels-1}]. Skipping this node.")
        except ValueError:
            print(f"Warning: Cannot convert Node ID {node_id_float} to integer. Skipping this node.")
    print(f"Successfully added {G.number_of_nodes()} nodes.")

    print("Adding edges to the graph...")
    edges_added_count = 0
    for index, row in graph_df.iterrows():
        try:
            src_node_idx = int(row['node_id_src'])
            dst_node_idx = int(row['node_id_dst'])
            length = row['branch_distance']

            if src_node_idx in G and dst_node_idx in G:
                branch_identifier = row['branch_id'] if 'branch_id' in graph_df.columns else index
                path_indices = skel_obj.path_coordinates(branch_identifier)
                path_indices = np.array(path_indices, dtype=int)

                if len(path_indices) > 0:
                    min_idx, max_idx = np.min(path_indices), np.max(path_indices)
                    if 0 <= min_idx and max_idx < num_skeleton_pixels:
                        path_coords = node_coords[path_indices]
                        G.add_edge(src_node_idx, dst_node_idx, weight=length, path=path_coords.tolist())
                        edges_added_count += 1
                    else:
                         print(f"Warning: Edge {src_node_idx}-{dst_node_idx} (branch ID: {branch_identifier}) path indices out of range [0, {num_skeleton_pixels-1}] (Range: [{min_idx}, {max_idx}]). Skipping this edge.")
                else:
                     print(f"Warning: Edge {src_node_idx}-{dst_node_idx} (branch ID: {branch_identifier}) has an empty path. Skipping this edge.")
            else:
                 print(f"Warning: Nodes for edge ({src_node_idx}-{dst_node_idx}) not found in graph. Skipping edge.")
        except (ValueError, TypeError) as e:
            print(f"Warning: Type or value error processing edge (index {index}): {e}. Skipping edge.")
        except (IndexError, KeyError) as e:
            print(f"Warning: Problem getting/processing path coordinates for edge (index {index}): {e}. Skipping edge.")

    print(f"Successfully added {edges_added_count} edges.")
    print("NetworkX graph construction complete.")
    print(f"Nodes in graph: {G.number_of_nodes()}")
    print(f"Edges in graph: {G.number_of_edges()}")

    # 6. Visualization
    if visualize:
        # --- Visualization for Non-Empty Skeleton ---
        print("Generating visualization...")

        # Use default font settings for English
        plt.style.use('default')
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img, cmap='gray', interpolation='nearest')
        ax[0].set_title('Original Mask Image') # English Title
        ax[0].axis('off')

        ax[1].imshow(skeleton_img, cmap='gray', interpolation='nearest')
        ax[1].set_title('Extracted Skeleton') # English Title
        ax[1].axis('off')

        ax[2].imshow(skeleton_img, cmap='gray', interpolation='nearest')
        ax[2].set_title('Topology Network Overlay') # English Title
        ax[2].axis('off')

        # Draw nodes and edges
        if not node_positions:
            print("Warning: No node position information available for plotting.")
        else:
            nx.draw_networkx_nodes(G, pos=node_positions, node_size=30, node_color='red', alpha=0.7, ax=ax[2])
            nx.draw_networkx_edges(G, pos=node_positions, edge_color='blue', alpha=0.7, width=1.5, ax=ax[2])

        plt.tight_layout()
        #plt.show()
        # --- End Visualization ---

    return G, img, skeleton_img

# --- Usage Example (Main execution block) ---
mask_path = 'labels'
json_output_path = 'top_json'
# Loop over all mask files in the directory
former_graph = None
for mask_file in os.listdir(mask_path):
    if mask_file.endswith('.jpg'):
        print(f"\nProcessing mask file: {mask_file}")
        image_file = os.path.join(mask_path, mask_file)
        output_graph_file_json = os.path.join(json_output_path, mask_file.replace('.jpg', '.json'))
        try:
            fracture_graph, original_image, skeleton_image = extract_fracture_network(image_file, threshold_value=128, visualize=True)

            if fracture_graph.number_of_nodes() > 0:
                print("\n--- Graph Information ---")
                print(f"Number of nodes: {fracture_graph.number_of_nodes()}")
                print(f"Number of edges: {fracture_graph.number_of_edges()}")

                # Calculate total length (optional)
                total_length = sum(d['weight'] for u, v, d in fracture_graph.edges(data=True))
                print(f"Estimated total fracture length (pixels): {total_length:.2f}")

                # Print info for an example node (optional)
                if fracture_graph.nodes:
                    example_node_id = list(fracture_graph.nodes())[0]
                    if example_node_id in fracture_graph:
                        node_data = fracture_graph.nodes[example_node_id]
                        print(f"\nInfo for node {example_node_id}:")
                        print(f"  Position (row, col): {node_data.get('pos', 'N/A')}")
                        print(f"  Degree (connected edges): {fracture_graph.degree[example_node_id]}")

                # Print info for an example edge (optional)
                if fracture_graph.edges:
                    example_edge = list(fracture_graph.edges(data=True))[0]
                    u, v, edge_data = example_edge
                    print(f"\nInfo for edge ({u}-{v}):")
                    print(f"  Length: {edge_data.get('weight', 'N/A'):.2f}")
                    # print(f"  Number of path points: {len(edge_data.get('path', []))}")

                # --- Save graph to JSON file ---
                try:
                    print(f"\nConverting graph to JSON Node-Link data and saving to: {output_graph_file_json}")
                    graph_data = nx.node_link_data(fracture_graph)
                    with open(output_graph_file_json, 'w', encoding='utf-8') as f:
                        json.dump(graph_data, f, indent=4)
                    print("Graph successfully saved as JSON.")
                except Exception as e:
                    print(f"Error: Failed to save graph to JSON: {e}")
                # --------------------------------
                # Calculate similarity index of label graph (optional)
                similarity_index_of_label_graph = similarity_index_of_label_graph_class()
                if former_graph is not None:
                    print("\nCalculating similarity index of label graph...")
                    similarity_index = similarity_index_of_label_graph(former_graph, fracture_graph)
                    print(f"Similarity index of label graph: {similarity_index:.4f}")
                former_graph = fracture_graph

            else:
                print("Graph has no nodes, skipping saving.")


        except FileNotFoundError as e:
            print(f"Error: Input image file not found at {image_file}")
            print(e)
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()