"""
Group Number 2
Marcus Kamen, Marius Schueller, Brynn LeBlanc, and Daniel Yakubu

To run this file you press play on main if using Pycharm. Otherwise, just run the main function with any
standard python interpreter. Make sure to have math, random, csv, queue, PriorityQueue, copy, re, and time
installed as packages on your computer for python. During runtime, you will be asked for the start city which should be
formatted in the form CityStateInitials (for example NashvilleTN), in addition to the file paths where the road trip
data will be obtained, the average speed you want to drive, and the file path to save the roadtrip. Our test run
output files are in sanfrancisco.txt, boston.txt, and nashville.txt

We decided to use an A* type search for our solution. The utility function combines both the preference
values of the nodes and edges as well as the distance of the traveled to node from the start location.
We used both of these values so that we could create a search that moves further away from the start for the
first half of the road trip, and moves closer to the start for the second half, while incorporating preference
values to help. We thought that this was a good strategy so that the road trip would go out from the start via
a high preferences, and then return to the start in a circular manner.
During our A* search we were able to use both path cost and heuristic distance by multiplying the preference
value of each node and edge by 50. We did this because the different edge values were between 21 and 338,
thereby putting the influence of preferences on being slightly less than the average distance, but still significant.
"""

import math
import random
import csv
from queue import PriorityQueue
import copy
import re
import time


class Node:
    """
        Class representation of a Node (a location in the road network)

        self.name       -- name of location
        self.x          -- latitude of location
        self.y          -- longitude of location
        self.preference -- preference values of location
    """

    def __init__(self, name, x, y):
        """
            Initialize instance of Node

            :param name: self.name copy
            :param x:    self.x copy
            :param y:    self.y copy
        """
        self.name = name
        self.x = x
        self.y = y
        self.preference = 0

    def time_at_location(self):
        """
            Get the time at this location based on the preference value. Increasing function
            that sets the 0 value at 5

            :return: time at this location
        """
        return self.preference * 100 + 5

    def __hash__(self):
        return hash(self.name)


class Edge:
    """
        Class representation of an edge between locations

        self.label          -- name of edge
        self.locationA      -- first location of edge
        self.locationB      -- second location of edge
        self.actualDistance -- distance between locations of edge
        self.preference     -- preference of edge
    """

    def __init__(self, label, locationA, locationB, actualDistance):
        """
            Initialize instance of edge

            :param label:          self.label copy
            :param locationA:      self.locationA copy
            :param locationB:      self.locationB copy
            :param actualDistance: self.actualDistance copy
        """
        self.label = label
        self.locationA = locationA
        self.locationB = locationB
        self.actualDistance = actualDistance
        self.preference = 0

    def add_time_on_edge(self, x):
        """
            Adds time for traveling edge (currently just calls time_at_location)

            :param x: speed to travel edge
            :return:  time at location for edge
        """
        return self.time_at_location(x)

    def time_at_location(self, x):
        """
            Gets the edge time at location
            :param x: speed to traverse edge
            :return: distance divided by speed
        """
        return self.actualDistance / x


class Roadtrip:
    """
        Class representation of a road trip (can be partial trip)

        self.NodeList           -- List of locations in trip
        self.EdgeList           -- List of edges in trip
        self.currentTimeElapsed -- Total time elapsed in trip
        self.time_search        -- Time in took to search this trip
        self.startNode          -- Start node of road trip
    """

    def __init__(self):
        """
            Initialize an instance of a Roadtrip, sets all fields to 0
        """
        self.NodeList = []
        self.EdgeList = []
        self.currentTimeElapsed = 0
        self.time_search = 0
        self.startNode = None

    def __lt__(self, other):
        """
            Overloaded less than operator for road trips
            Needed in case library functions use less than operators to break ties between equal utilities
            in PriorityQueue

            :param other: other Roadtrip working with
            :return: boolean if self < other
        """
        return self.total_preference() < other.total_preference()

    def total_preference(self):
        """
            Gets the total preference of all nodes and edges in Roadtrip

            :return: total preference
        """
        visited = set()
        preference = 0.0
        for node in self.NodeList:
            if node != self.startNode and node not in visited:
                preference = preference + node.preference
                visited.add(node)

        for edge in self.EdgeList:
            if edge not in visited:
                preference = preference + edge.preference
                visited.add(edge)

        return preference

    def get_total_distance(self):
        """
            Gets the total distance traveled on the road trip

            :return: sum of distances of all edges
        """
        return sum(edge.actualDistance for edge in self.EdgeList)

    # get time estimate of trip
    def time_estimate(self, x):
        """
            Gets the time estimate for the full Roadtrip
            

            :param x: speed to travel edges
            :return: total time of Roadtrip
        """
        visited = set()
        time = 0.0
        for node in self.NodeList:
            if node != self.startNode and node not in visited:
                time = time + node.time_at_location()
                visited.add(node)

        for edge in self.EdgeList:
            if edge not in self.EdgeList:
                time = time + edge.add_time_on_edge(x)
                visited.add(edge)

        return time

    def hasNode(self, node):
        """
            Checks if a node is present in this Roadtrip

            :param node: node to check
            :return: if node is present in NodeList
        """
        for nod in self.NodeList:
            if nod.name == node.name:
                return True
        return False

    def get_node_by_location(self, node_name):
        """
            Gets a node based on its name

            :param node_name: name of node
            :return: node with that name
        """
        for node in self.NodeList:
            if node.name == node_name:
                return node
        raise ValueError(f"No node {node_name} in road trip network")

    def find_NodeB(self, edge):
        """
            Finds the second node listed in an edge

            :param edge: Edge to check
            :return: Node corresponding to second node of edge
        """
        for node in self.NodeList:
            if edge.locationB == node.name:
                return node

    def find_NodeA(self, edge):
        """
            Finds the first node listed in an edge

            :param edge: Edge to check
            :return: Node corresponding to first node of edge
        """
        for node in self.NodeList:
            if edge.locationA == node.name:
                return node
            
    def get_theme_count_data_and_labels(self):
        attractions_mapping = LoadThemesFromFile('Road network - Attractions.csv')
        theme_count = {}
        for themes_list in attractions_mapping.values():
            for theme in themes_list:
                if theme in theme_count:
                    theme_count[theme] += 1
                else:
                    theme_count[theme] = 1

        # Extracting themes and their counts
        labels = list(theme_count.keys())
        data = list(theme_count.values())

        return labels, data


    def print_result(self, num, start_node, maxTime, speed_in_mph):
        """
            Print the results of a road trip.

            :param start_node: (node or string) The starting node or location. If a string is provided,
                                it will be used to fetch the corresponding Node using get_node_by_location.
            :param maxTime: (float) The maximum time allowed for the trip.
            :param speed_in_mph: (float) The speed in miles per hour used for time estimation.
            :return: None

            Prints the simulation results, including routing details and summary information,
            to the console. The output includes the starting node, maximum time, speed, routing
            details, and summary information such as total preference, total distance, and
            estimated time.

            Example:
            ```
            router = YourRouterClass()
            router.print_result("StartLocation", 10.0, 60.0)
            ```
        """
        if not isinstance(start_node, Node):
            start_node = self.get_node_by_location(start_node)

        cur_node = start_node
        line_number = 1

        print(f"Solution {num}", end=" ")
        print(start_node.name, end=" ")
        print(maxTime, end=" ")
        print(speed_in_mph, end=" ")
        print("\n")

        for edge in self.EdgeList:

            print(line_number, ".", end=" ")

            print(cur_node.name, end=" ")

            if self.find_NodeA(edge) == cur_node:
                cur_node = self.find_NodeB(edge)
            else:
                cur_node = self.find_NodeA(edge)

            print(cur_node.name, end=" ")
            print(edge.label, end=" ")
            print(edge.preference, end=" ")
            print(edge.add_time_on_edge(speed_in_mph), end=" ")
            print(cur_node.preference, end=" ")
            print(cur_node.time_at_location(), end=" ")
            print("\n")
            line_number += 1

        print(start_node.name, end=" ")
        print(self.total_preference(), end=" ")
        print(self.get_total_distance(), end=" ")
        print(self.time_estimate(speed_in_mph), end=" ")

    def write_result_to_file(self, num, start_node, maxTime, speed_in_mph, output_file=None):
        """
            Write the results of a round trip to a file.

            :param start_node: (Node or string) The starting node or location. If a string is provided,
                                                it will be used to fetch the corresponding Node using
                                                get_node_by_location.
            :param maxTime: (float) The maximum time allowed for the trip.
            :param speed_in_mph: (float) The speed in miles per hour used for time estimation.
            :param output_file: (str, optional) The name of the file to write the results to.
                                If not provided, the default filename is "default_output.txt".
            :return: None

            Writes the simulation results, including routing details and summary information,
            to the specified output file. The file includes the starting node, maximum time,
            speed, routing details, and summary information such as total preference and
            estimated time.

            Example:
            ```
            router = YourRouterClass()
            router.write_result_to_file("StartLocation", 10.0, 60.0, "output.txt")
            ```
        """
        if not isinstance(start_node, Node):
            start_node = self.get_node_by_location(start_node)

        cur_node = start_node
        line_number = 1

        if output_file is None:
            output_file = "default_output.txt"

        with open(output_file, 'a', encoding='utf-8') as file:
            file.write(f"Solution{num} ")
            file.write(f"{start_node.name} ")
            file.write(f"{maxTime} ")
            file.write(f"{speed_in_mph} ")
            file.write("\n")

            for edge in self.EdgeList:
                file.write(f"{line_number}. ")
                file.write(f"{cur_node.name} ")

                if self.find_NodeA(edge) == cur_node:
                    cur_node = self.find_NodeB(edge)
                else:
                    cur_node = self.find_NodeA(edge)

                file.write(f"{cur_node.name} ")
                file.write(f"{edge.label} ")
                file.write(f"{edge.preference} ")
                file.write(f"{edge.add_time_on_edge(speed_in_mph)}")
                file.write(f"{cur_node.preference} ")
                file.write(f"{cur_node.time_at_location()} ")
                file.write("\n")
                line_number += 1

            file.write(f"{start_node.name} ")
            file.write(f"{self.total_preference()} ")
            file.write(f"{self.get_total_distance()} ")
            file.write(f"{self.time_estimate(speed_in_mph)} ")
            file.write("\n")
            file.write("\n")
            

class RegressionTree:
    """
        Representation of a regression tree for utilities
        
        self.root() -- root of tree (tree is recursively defined)
    """
    
    def __init__(self):
        """
            Initialize regression tree
        """
        self.root = RegressionNode(None, None, None, None, 0.5)
        
    def traverse_node(self, sample, node):
        """
            Return the correct leaf node
        """
        if node.is_leaf_node():
            return node
        
        # If the current node is not a leaf node, traverse further based on the sample's feature value
        if sample[node.feature] <= node.threshold:
            # If the sample's feature value is less than or equal to the node's threshold, traverse left
            return self.traverse_tree(sample, node.left)
        else:
            # If the sample's feature value is greater than the node's threshold, traverse right
            return self.traverse_tree(sample, node.right)

    def traverse_result(self, sample, node):
        """
            Return the correct node value
        """
        if node.is_leaf_node():
            return node.value
        
        # If the current node is not a leaf node, traverse further based on the sample's feature value
        if sample[node.feature] <= node.threshold:
            # If the sample's feature value is less than or equal to the node's threshold, traverse left
            return self.traverse_tree(sample, node.left)
        else:
            # If the sample's feature value is greater than the node's threshold, traverse right
            return self.traverse_tree(sample, node.right)
        

    def split(self, feature, sample, split_val, utility1, utility2):
        """
            Add nodes to the regression tree
        """
        loc = self.traverse_node(sample, self.root)
        loc.value = None
        loc.feature = feature
        loc.threshold = split_val
        loc.left = RegressionNode(None, None, None, None, utility1)
        loc.right = RegressionNode(None, None, None, None, utility2)
        
        


class RegressionNode:
    """
    A class representing a node in a decision tree.

    Attributes:
    - feature (int or None): The index of the feature used for splitting at this node.
    - threshold (float or None): The threshold value used for splitting the feature.
    - left (Node or None): The left child node.
    - right (Node or None): The right child node.
    - value (int or None): The class label assigned to this node if it is a leaf node.

    Methods:
    - is_leaf_node(): Returns True if the node is a leaf node, False otherwise.

    The Node class represents a node in a decision tree. Each node contains information about
    the splitting feature, threshold, child nodes, and assigned class label (if it's a leaf node).

    Constructor Parameters:
    - feature (int or None): Index of the feature used for splitting at this node.
    - threshold (float or None): Threshold value used for splitting the feature.
    - left (Node or None): Left child node.
    - right (Node or None): Right child node.
    - value (int or None, optional): Class label assigned to this node if it's a leaf node.

    If the value parameter is provided, it indicates that the node is a leaf node, and the
    decision tree stops splitting further.

    Example:
    >>> node = Node(feature=0, threshold=2.5, left=Node(value=1), right=Node(value=0))
    >>> node.is_leaf_node()
    False
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
        - bool: True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class Roadtripnetwork:
    """
            Representation of a Roadtripnetwork object with information about the road network,
            including start location, file paths, maximum time allowed, speed, and result file.

            self.NodeList   -- List of nodes in network
            self.EdgeList   -- List of edges in network
            self.startLoc   -- Start location of search
            self.LocFile    -- File to find locations
            self.EdgeFile   -- File to find edges
            self.maxTime    -- Max time of road trip
            self.x_mph      -- Time to traverse an edge
            self.resultFile -- Where to output results
            self.startNode  -- Node corresponding to self.startLoc
            self.solutions  -- Road trip solutions currently found
            self.max_trials -- Maximum number of road trips the user wants to find
            self.forbidden_locations        -- Locations forbidden by the user
            self.required_locations         -- Locations required by the user
            self.index_in_required_checked  -- Variable to determine which required locations have been visited so far
            self.next_required_node         -- Variable to determine which required location will be attempted to be visited next
            self.Regression_tree
    """

    def __init__(self, startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile, max_trials, forbidden_locations, required_locations):
        """
            Initialize a Roadtripnetwork object

            :param startLoc:    self.startLoc copy
            :param LocFile:     self.LocFile copy
            :param EdgeFile:    self.EdgeFile copy
            :param maxTime:     self.maxTime copy
            :param x_mph:       self.x_mph copy
            :param resultFile:  self.resultList copy
            :param max_trails:  self.max_trails copy
            :param forbidden:   self.forbidden_locations copy
            :param required:    self.required_locations copy
        """
        self.NodeList = []
        self.EdgeList = []
        self.startLoc = startLoc
        self.LocFile = LocFile
        self.EdgeFile = EdgeFile
        self.maxTime = maxTime
        self.x_mph = x_mph
        self.resultFile = resultFile
        self.startNode = None
        self.solutions = PriorityQueue()
        self.max_trials = max_trials
        self.forbidden_locations = forbidden_locations
        self.required_locations = required_locations
        self.index_in_required_checked = 0
        self.next_required_node = None
        self.regression_tree = None

    def location_preference_assignments(self, a=0.0, b=1.0):
        """
                Assign random preferences to all nodes in the road network within a specified range.

                :param a: Lower bound of the preference range.
                :param b: Upper bound of the preference range.
        """
        for node in self.NodeList:
            node.preference = random.uniform(a, b)

    def edge_preference_assignments(self, a=0.0, b=0.1):
        """
                Assign random preferences to all edges in the road network within a specified range.

                :param a: Lower bound of the preference range.
                :param b: Upper bound of the preference range.
        """
        for edge in self.EdgeList:
            edge.preference = random.uniform(a, b)

    def parseNodes(self):
        """
            Parse nodes from the CSV file and create Node objects for each location in the road network.
        """

        file_path = self.LocFile

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location_info = {
                    'Location Label': row['Location Label'],
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                }

                # USE NODE CLASS
                self.NodeList.append(Node(location_info['Location Label'], location_info['Latitude'],
                                          location_info['Longitude']))

    def parseEdges(self):
        """
            Parse edges from the CSV file and create Edge objects for each connection in the road network.
        """

        file_path = self.EdgeFile  # Replace with your actual file path

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location_info = {
                    'edgeLabel': row['edgeLabel'],
                    'locationA': row['locationA'],
                    'locationB': row['locationB'],
                    'actualDistance': float(row['actualDistance']),
                }

                # USE EDGE CLASS
                self.EdgeList.append(Edge(location_info['edgeLabel'], location_info['locationA'],
                                          location_info['locationB'], location_info['actualDistance']))

    def loadFromFile(self):
        """
            Loads data from files calling parseNodes() and parseEdges()
        """
        self.parseNodes()
        self.parseEdges()


    def initializeForSearch(self, tree):
        """
            Initializes the start node and assigns preferences before starting the search algorithm
            
            :param tree: The Regression tree to intialize (1 or 2)
        """
        # initialize correct regression tree
        
        self.location_preference_assignments()
        self.edge_preference_assignments()

        for node in self.NodeList:
            if self.startLoc == node.name:
                self.startNode = node
                break
        
        if self.index_in_required_checked < len(self.required_locations) and not self.required_locations[self.index_in_required_checked] == ' ':
            for node in self.NodeList:
                if self.required_locations[0] == node.name:
                    self.next_required_node = node
                    break
                

    def astar_search(self):
        """
            Perform A* search to find an optimal path considering preferences and distances as evenly as possible.
            Updates the Roadtrip with the discovered path.
        """

        search_start = time.time()
        numSearches = 0
        frontier = PriorityQueue()
        trip = Roadtrip()
        trip.NodeList.append(self.startNode)
        trip.startNode = self.startNode
        frontier.put((0, trip))

        while numSearches < self.max_trials and (not frontier.empty()):
            trip = frontier.get()[1]
            # check if start node returned to
            if len(trip.EdgeList) > 1:
                if (self.find_NodeA(trip.EdgeList[len(trip.EdgeList) - 1]) == self.startNode
                        or self.find_NodeB(trip.EdgeList[-1]) == self.startNode):
                    trip.time_search = time.time() - search_start
                    self.solutions.put((-trip.total_preference(), trip))
                    numSearches = numSearches + 1
                    search_start = time.time()
                    continue

            # go through edge list and find related nodes
            for edge in self.EdgeList:
                name = trip.NodeList[len(trip.NodeList) - 1].name
                if edge.locationA == name:
                    node = self.find_NodeB(edge)
                    util = self.utility(trip, edge, node)
                    if not util == float('inf'):
                        newTrip = copy.deepcopy(trip)
                        newTrip.NodeList.append(node)
                        newTrip.EdgeList.append(edge)
                        frontier.put((util, newTrip))
                elif edge.locationB == name:
                    node = self.find_NodeA(edge)
                    util = self.utility(trip, edge, node)
                    if not util == float('inf'):
                        newTrip = copy.deepcopy(trip)
                        newTrip.NodeList.append(node)
                        newTrip.EdgeList.append(edge)
                        frontier.put((util, newTrip))

    def utility(self, trip, edge, node):
        """
            Calculate the utility value for a given edge and node based on distance, preferences, and time constraints.

            :param trip: Trip object
            :param edge: Edge object representing the road segment.
            :param node: Node object representing the location.
            :return: Utility value considering distance to the start, node and edge preferences.
        """
        
        # use Regression tree for utilities instead
        
        timeEstimate = trip.time_estimate(self.x_mph)
        
        if self.index_in_required_checked < len(self.required_locations) and not self.required_locations[self.index_in_required_checked] == '':
            distToLocation = math.sqrt(math.pow(node.x - self.next_required_node.x, 2) + math.pow(node.y - self.next_required_node.y, 2))
            
            if node.name in self.forbidden_locations or timeEstimate > self.maxTime:
                return float('inf')
            
            if node.name == self.required_locations[self.index_in_required_checked]:
                self.index_in_required_checked = self.index_in_required_checked + 1
                if self.index_in_required_checked < len(self.required_locations) and not self.required_locations[self.index_in_required_checked] == ' ':
                    for node in self.NodeList:
                        if self.required_locations[self.index_in_required_checked] == node.name:
                            self.next_required_node = node
                            break
                return -float('inf')
            
            if timeEstimate < self.maxTime:
                distToLocation = distToLocation * (-1)
            else:
                return float('inf')

            if trip.hasNode(node):
                return distToLocation + 10
            return distToLocation - 50 * node.preference - 50 * edge.preference


        distToStart = math.sqrt(math.pow(node.x - self.startNode.x, 2) + math.pow(node.y - self.startNode.y, 2))
        # If the node is in the forbidden locations, return infinite utility to avoid selecting it
        if node.name in self.forbidden_locations or timeEstimate > self.maxTime:
            return float('inf')
        
        if node.name in self.required_locations and not trip.hasNode(node):
            return -float('inf')

        # Adjust distance to start based on whether the trip is in the first half or the second half of the max time
        if timeEstimate < self.maxTime / 2:
            distToStart = distToStart * (-1)
        else:
            distToStart = distToStart - 1000

        if trip.hasNode(node):
            return distToStart + 10
        return distToStart - 50 * node.preference - 50 * edge.preference

    def find_NodeB(self, edge):
        """
            Find the node associated with location B on an edge

            :param edge: Edge to check
            :return: Node in location B of edge
        """
        for node in self.NodeList:
            if edge.locationB == node.name:
                return node

    def find_NodeA(self, edge):
        """
            Find the node associated with location A on an edge

            :param edge: Edge to check
            :return: Node in location A of edge
        """

        for node in self.NodeList:
            if edge.locationA == node.name:
                return node


def RoundTripRoadTrip(startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile, max_trials, forbidden_locations, required_locations, tree):
    """
        Perform a round-trip road trip optimization using the A* search algorithm.

        :param startLoc: Starting location for the road trip.
        :param LocFile: File path containing location data (CSV format).
        :param EdgeFile: File path containing road network data (CSV format).
        :param maxTime: Maximum allowable time for the road trip in minutes.
        :param x_mph: Speed in miles per hour for estimating travel times.
        :param resultFile: File path to save the optimization result.
        :param max_trials: Number of road trips to create and print to user
        :param forbidden_locations: Locations that the user does not want to visit
        :param required_locations: Locations that the user must visit
        :param tree
    """
    locsAndRoads = Roadtripnetwork(startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile, max_trials, forbidden_locations, required_locations)
    locsAndRoads.loadFromFile()
    locsAndRoads.initializeForSearch(tree)
    locsAndRoads.astar_search()
    return locsAndRoads.solutions

def add_suffix(filename, suffix):
    """
        Adds suffix to filename to correctly use file

        :param filename: filename to add to
        :param suffix: suffix to add
        :return: filename with suffix appended
    """

    # Split the filename into base and extension
    base, extension = filename.rsplit('.', 1)

    # Append the suffix and reassemble the filename
    new_filename = f"{base}{suffix}.{extension}"

    return new_filename

def LoadThemesFromFile(attractions_csv_file):
        """
        Load attraction themes from a CSV file into a dictionary.

        Args:
            attractions_csv_file (str): The path to the CSV file containing attraction data.
        
        Returns:
             dict: A dictionary where keys are attraction locations or edges, and values are lists of themes.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
        """
        attraction_theme_mapping = {}
        try:
            with open(attractions_csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    loc_or_edge = row['Loc or Edge Label']
                    themes = row['Themes'].split(', ')
                    attraction_theme_mapping[loc_or_edge] = themes
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{attractions_csv_file}' does not exist in specified directory.")
        return attraction_theme_mapping


def checkLists(required, forbidden):
    for place in required:
        for place2 in forbidden:
            if place == place2:
                return False
    return True




def main():
    """
        Run program
    """

    num_trials = 1
    print("Welcome to RoundTrip Recommender! Please enter details about your round trip")
    print("If you do not want to specify any of the entries, just click enter and a default value will be used.")
    start_location = input("Enter the starting location for the road trip: ") or "NashvilleTN"

    no_duplicates = False
    while not no_duplicates:
        required_locations = input("Enter any locations that must be a part of your trip (separated by \", \"). Note that all locations must be valid:") or ""
        required_locations_list = required_locations.split(", ")
        forbidden_locations = input("Enter any locations that you do not want to be a part of your trip (separated by "
                                    "\", \"). Note that all locations must be valid:") or " "
        forbidden_locations_list = forbidden_locations.split(", ")
        no_duplicates = checkLists(required_locations_list, forbidden_locations_list)
        if not no_duplicates:
            print("Same cities found in forbidden and required locations")
            print("Please re-enter cities")

    """
    option for soft forbidden location
    """
    location_file = input(
        "Enter the file path containing location data (CSV format): ") or "Road Network - Locations.csv"
    edge_file = input("Enter the file path containing road network data (CSV format): ") or "Road Network - Edges.csv"
    max_time = int(input("Enter the maximum allowable time for the road trip: ") or 540)
    speed_in_mph = int(input("Enter the speed in miles per hour for estimating travel times: ") or 60)
    result_file = input("Enter the file path to save the road trip result: ") or "result.txt"
    max_trials = int(input("Enter the maximum number of road trips you would like to display: ") or 3)
    tree = int(input("Enter the regression tree you would like to use (1/2): ") or 1)
    
    round_trips = RoundTripRoadTrip(start_location, location_file, edge_file, max_time, speed_in_mph, result_file, max_trials, forbidden_locations_list, required_locations_list, tree)

    runtimes = []
    preferences = []

    first_trip = round_trips.get()
    first_trip[1].print_result(num_trials, start_location, max_time, speed_in_mph)
    first_trip[1].write_result_to_file(num_trials, start_location, max_time, speed_in_mph, result_file)
    num_trials += 1

    runtimes.append(first_trip[1].time_search)
    preferences.append(first_trip[1].total_preference())

    while num_trials <= max_trials:
        go_again = input(f"\nDo you want to print your next road trip (printed {num_trials - 1} of {max_trials} trips created)? (yes/no): ").lower()
        if go_again != 'yes':
            break
        else:
            cur_trip = round_trips.get()
            cur_trip[1].print_result(num_trials, start_location, max_time, speed_in_mph)
            cur_trip[1].write_result_to_file(num_trials, start_location, max_time, speed_in_mph, result_file)
            num_trials += 1
            runtimes.append(cur_trip[1].time_search)
            preferences.append(cur_trip[1].total_preference())

    average_runtime = sum(runtimes) / len(runtimes)
    average_preference = sum(preferences) / len(preferences)
    max_preference = preferences[0]
    min_preference = preferences[0]

    for val in preferences:
        if val > max_preference:
            max_preference = val
        if val < min_preference:
            min_preference = val

    print("\n\nSummary of Output")
    print(f"Average runtime of searches: {average_runtime}")
    print(f"Maximum trip preference: {max_preference}")
    print(f"Average trip preference: {average_preference}")
    print(f"Minimum trip preference: {min_preference}")

    with open(result_file, 'a', encoding='utf-8') as file:
        file.write("Summary of Output\n")
        file.write(f"Average runtime of searches: {average_runtime}\n")
        file.write(f"Maximum trip preference: {max_preference}\n")
        file.write(f"Average trip preference: {average_preference}\n")
        file.write(f"Minimum trip preference: {min_preference}\n")


if __name__ == '__main__':
    main()

"""
In general, there is no solution for a road trip in which the starting location is not on the list of locations in the
provided csv file. For instance, if you wanted to start your trip in a small town called SolonOH, this will not work
because it is not in the csv file provided. In addition to this, the program does not run more than 3 times for one
starting location, so if a user wanted more than 3 road trip options, this would not work. Finally, if the number of
allotted hours is very small or very large (10 or 100000 hours), when the speed is very slow (10mphs) the program will
either not find a route, or take a very long time to run. In general, the amount of time for each road trip is not
strictly under the time limit, but have a range of around +- 50 hours from the given time alloted for the trip.
In addition to this, the amount of time as well as the preference for the road trips generally decreases between each
suggested trip.

Average runtime of all searches for all test runs: 0.131
Average maximum trip preference for all test runs: 5.723
Average total tip preference for all test runs:    5.426
Average minimum trip preference for all test runs: 5.129
"""
