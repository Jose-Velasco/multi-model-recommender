from typing import Callable
from torch_geometric.data import Data, InMemoryDataset
from typing import Any
from pathlib import Path
import torch
from functools import reduce
from functools import partial

class PositiveIterationGraph(InMemoryDataset):
    def __init__(self,
                 root: str | None = None, 
                 transform: Callable[..., Any] | None = None,
                 pre_transform: Callable[..., Any] | None = None,
                 pre_filter: Callable[..., Any] | None = None,
                 log: bool = True, 
                 force_reload: bool = False,
                 raw_dataset_txt_path: str = "/train-1.txt",
                 is_undirected: bool = True,
                 add_user_item_node_mapping: bool = False,
                 bipartite_pre_transform: list[partial] | None = None):
        self.raw_dataset_txt_path = raw_dataset_txt_path
        self.is_undirected = is_undirected
        self.add_user_item_node_mapping = add_user_item_node_mapping
        self._processing_node_idx_counter: int = 0
        self.bipartite_pre_transform = bipartite_pre_transform
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        # return [self.raw_dataset_txt_path]
        return [f"{self.root}/raw/{self.raw_dataset_txt_path}"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        """
        data comes from course canvas
        """
        # Download to `self.raw_dir`.
        pass

    def process(self):
        file_path = Path(self.raw_file_names[0])
        # users and items are nodes in the graph
        # mapping user_ids to a node_id
        # keys: user_id, values: user_node_id
        # node_id = 0
        users_ids: dict[int, int] = {}
        # keys: item_id, values: item_node_id
        item_ids: dict[int, int] = {}
        # shape [2, num_edges]
        # "top" list is source node, "bottom" list is destination node
        # edges are based on index position in 
        user_item_edges: list[list] = [
            [],
            []
        ]
        with file_path.open("r") as f:
            # in each row first value id user_id and the rest are item user interacted with
            for line in f:
                self._process_line(line=line, users_ids=users_ids, item_ids=item_ids, user_item_edges=user_item_edges)
        # nodes = torch.zeros(len(users_ids) + len(item_ids), 1, dtype=torch.float32)
        nodes = torch.ones(len(users_ids) + len(item_ids), 1, dtype=torch.float32)
        edges = torch.tensor(user_item_edges, dtype=torch.int64)
        graph = Data(
            x=nodes,
            edge_index=edges
        )
        if self.add_user_item_node_mapping:
            graph.users_node_id_map = users_ids
            graph.item_node_id_map = item_ids

        if self.pre_filter is not None:
            graph = self.pre_filter(graph)

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if self.add_user_item_node_mapping and self.bipartite_pre_transform is not None:
            user_node_id_inv_map = {v: k for k, v in graph.users_node_id_map.items()}
            graph = reduce(lambda g, transformation: transformation(user_node_id_inv_map)(g), self.bipartite_pre_transform, graph)
        
        # graph.num_nodes = len(users_ids) + len(item_ids)
        self.save(data_list=[graph], path=self.processed_paths[0])

    def _process_line(self, line: str, users_ids: dict[int, int], item_ids: dict[int, int], user_item_edges: list[list]) -> None:
        line_list: list[int] = [int(entity_id) for entity_id in line.split()]
        current_user_id = line_list[0]
        self._insert_into_entity_id_node_id_map(entity_id=current_user_id, entity_to_node_id_map=users_ids)
        for item_id in line_list[1:]:
            if item_id not in item_ids:
                # add assigns item_id to a node_id
                self._insert_into_entity_id_node_id_map(entity_id=item_id, entity_to_node_id_map=item_ids)

            # appends the current users node id as source node
            # appends the current item node id as destination node
            self._add_edge(source=users_ids[current_user_id], destination=item_ids[item_id], edge_index=user_item_edges)
            
            # if want the graph to be undirected revere edge is added
            # currently directed graph version is from users to items TODO: make reverse toggleable
            if self.is_undirected:
                self._add_edge(source=item_ids[item_id], destination=users_ids[current_user_id], edge_index=user_item_edges)
    
    def _insert_into_entity_id_node_id_map(self, entity_id: int, entity_to_node_id_map: dict[int, int]) -> None:
        if entity_id in entity_to_node_id_map:
            raise KeyError(f"Error: user/item_id '{entity_id}' already has a user/item_node_id of {entity_to_node_id_map[entity_id]}. Possible duplicate user/item.")
        entity_to_node_id_map[entity_id] = self._processing_node_idx_counter
        self._processing_node_idx_counter += 1

    def _add_edge(self, source: int, destination: int, edge_index: list[list]) -> None:
        # appends the source node id as source node
        edge_index[0].append(source)
        # appends the destination node id as destination node
        edge_index[1].append(destination)
