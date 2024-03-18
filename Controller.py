# import torch
from dataclasses import field
import numpy as np
import math
import heapq


class Node:
    x: int = field(compare=False)
    y: int = field(compare=False)
    distance: float
    visited: bool = field(compare=False)
    previous_node: list = field(compare=False)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = math.inf
        self.visited = False
        self.previous_node = [None, None]

    def __lt__(self, node):
        return self.distance < node.distance


class Controller:

    def __init__(self, height, width):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = height
        self.width = width
        # self.all_actions = np.array([[0, 0],
        #                              [1, 0], [-1, 0], [0, 1], [0, -1]])

        self.all_actions = np.array([[0, 0],
                                     [1, 0], [-1, 0], [0, 1], [0, -1],
                                     [1, 1], [-1, -1], [-1, 1], [1, -1]])

        # self.action_id_dict = {str(self.all_actions[i]): i for i in range(self.all_actions.shape[0])}

    def get_node_neighbours(self, x, y):
        neighbours = []
        # loc = np.array([x, y])
        # for action in self.all_actions:
        #     if np.all(action == np.array([0, 0])):
        #         continue
        #     if np.all(loc + action >= 0) and np.all(loc + action < [self.height, self.width]):
        #         neighbours.append(loc + action)
        if x - 1 >= 0:
            neighbours.append([x - 1, y])
            if y - 1 >= 0:
                neighbours.append([x - 1, y - 1])
            if y + 1 < self.width:
                neighbours.append([x - 1, y + 1])
        if x + 1 < self.height:
            neighbours.append([x + 1, y])
            if y - 1 >= 0:
                neighbours.append([x + 1, y - 1])
            if y + 1 < self.width:
                neighbours.append([x + 1, y + 1])
        if y - 1 >= 0:
            neighbours.append([x, y - 1])
        if y + 1 < self.width:
            neighbours.append([x, y + 1])
        return neighbours

    def get_shortest_path_to_object(self, source, target):
        graph = np.empty((self.height, self.width), dtype=object)
        min_heap = []
        heapq.heapify(min_heap)

        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                graph[i, j] = Node(i, j)
                if i == source[0, 0] and j == source[0, 1]:
                    graph[i, j].distance = 0
                heapq.heappush(min_heap, graph[i, j])

        while len(min_heap) > 0:
            at = heapq.heappop(min_heap)
            if at.visited:
                continue
            at.visited = True
            neighbours = self.get_node_neighbours(at.x, at.y)
            for node in neighbours:
                new_edge = at.distance + math.dist([at.x, at.y], node)
                if new_edge < graph[node[0], node[1]].distance:
                    graph[node[0], node[1]].distance = new_edge
                    graph[node[0], node[1]].previous_node = [at.x, at.y]

            heapq.heapify(min_heap)

        # target = target.numpy()
        # distance_to_targets = np.array([graph[at[0], at[1]].distance for at in target])
        at = target[0] #target[np.argmin(distance_to_targets)]
        actions = []
        while not (at[0] == source[0, 0] and at[1] == source[0, 1]):
            actions.append(at - np.array(graph[at[0], at[1]].previous_node))
            at = graph[at[0], at[1]].previous_node
        return actions

    def get_action(self, source_location, target_location):
        actions = self.get_shortest_path_to_object(source_location, target_location)
        if len(actions) > 0:
            return actions
        else:
            return np.expand_dims(self.all_actions[0], axis=0)
