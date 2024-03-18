"""""
Brynn LeBlanc, Daniel Yabuku, Markus Kamen, Marius Schueller
Group 2

"""

import math
import pandas as pd
import numpy as np
from collections import Counter


class Node:
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

class DecisionTreeClassifier:

    """
    A decision tree classifier.

    Parameters:
    -----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node. If fewer than min_samples_split 
        samples are available, the node is not split further, and it becomes a leaf node.

    max_depth : int, default=5
        The maximum depth of the decision tree. The tree will be pruned after reaching this depth to control
        model complexity and prevent overfitting.

    n_features : int or None, default=None
        The number of features to consider when looking for the best split. If None, all features will be considered.
        This parameter can be useful for controlling the randomness of the decision tree, especially when used in 
        ensemble methods like random forests.

    impurity_function: String or functiom
        The function you want to use for impurity calculations to determine splits. You can use
        either information_gain or gini_index. You can also pass in a custom function.
        The custom function should have the following signature:
        custom_func(self, training_labels, training_column, split_threshold) and return an integer

     root : Node
        The root node of the decision tree.
    """

    def __init__(self, min_samples_split=2, max_depth=5, impurity_function="information_gain", n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.impurity_mapping = {
            "information_gain": self.information_gain,
            "gini_gain": self.gini_gain
        }
        self.impurity_function = self.impurity_mapping[impurity_function] if not callable(impurity_function) else impurity_function

    def fit (self, training_data, training_labels):

        """
        Fit the decision tree classifier to the training data.

        Parameters:
        -----------
        training_data : array-like of shape (n_samples, n_features)
            The training data samples. Each row represents a sample, and each column represents a feature.

        training_labels : array-like of shape (n_samples,)
            The target labels corresponding to the training data samples.

        Returns:
        --------
        self : DecisionTreeClassifier
            Returns the fitted DecisionTreeClassifier instance.

        Notes:
        ------
        The decision tree is built using the provided training data and labels. The tree is constructed
        recursively by recursively splitting the data based on the features to minimize impurity or maximize
        information gain.
        """
        #Build the tree

        self.n_features = training_data.shape[1] if not self.n_features else min(self.n_features, training_data.shape[1])
        self.root = self.grow_tree(training_data, training_labels)


    def grow_tree(self, training_data, training_labels, depth=0):
        """
        Recursively grows the decision tree.

        Parameters:
        -----------
        training_data : array-like of shape (n_samples, n_features)
            The training data samples. Each row represents a sample, and each column represents a feature.

        training_labels : array-like of shape (n_samples,)
            The target labels corresponding to the training data samples.

        depth : int, default=0
            The current depth of the tree. Used to enforce the maximum depth constraint.

        Returns:
        --------
        node : Node
            The root node of the decision subtree constructed during this recursive step.

        Notes:
        ------
        This method recursively constructs a decision tree based on the provided training data and labels.
        It stops growing the tree when one of the stopping criteria is met: maximum depth reached, only one
        class label remains, or the number of samples is less than the minimum samples required for split.
        """

        # Get the number of samples and features
        n_samples, n_features = training_data.shape

        # Get the number of unique labels
        n_labels = len(np.unique(training_labels))

        # Stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):

            # Create a leaf node with the most common label
            leaf_value = self.most_common_label(training_labels)
            return Node(value=leaf_value)
        
        # Randomly select feature indices
        # This line randomly selects a subset of feature indices from the total number of features available in the dataset.
        # It is used to consider only a subset of features during each split of the decision tree, which can help improve model robustness 
        # and generalization and thus prevent overfitting. However, it is likely not doing much work in this
        # case since there are a limited number of features. Still, we include it for the sake of it.
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Greedy search for the best split
        best_feature, best_threshold = self.best_criteria(training_data, training_labels, feature_indices)

        # Split the data based on the best feature and threshold
        left_indices, right_indices = self.split(training_data[:, best_feature], best_threshold)

        # Recursively grow the left and right subtrees
        left = self.grow_tree(training_data[left_indices, :], training_labels[left_indices], depth + 1)
        right = self.grow_tree(training_data[right_indices, :], training_labels[right_indices], depth + 1)

        # Create a node for the best split
        return Node(best_feature, best_threshold, left, right)

    def best_criteria(self, training_data, training_labels, feature_indices):
        """
        Find the best split criteria for the decision tree.

        Parameters:
        -----------
        training_data : array-like of shape (n_samples, n_features)
            The training data samples. Each row represents a sample, and each column represents a feature.

        training_labels : array-like of shape (n_samples,)
            The target labels corresponding to the training data samples.

        feature_indices : array-like of shape (n_features,)
            The indices of the features to consider for splitting the data.

        Returns:
        --------
        split_index : int
            The index of the feature that provides the best split.

        split_threshold : float
            The threshold value for the best split.

        Notes:
        ------
        This method iterates through all possible splits for the given features and calculates the information
        gain for each split. It returns the feature index and threshold value that result in the highest information
        gain, indicating the best split criteria for the decision tree.
        """
        # Initialize variables to track the best gain and corresponding split criteria
        best_gain = -1
        split_index, split_threshold = None, None

        # Iterate through each feature index
        for feature_index in feature_indices:
            # Extract the column corresponding to the current feature
            training_column = training_data[:, feature_index]

            # Find unique thresholds for the current feature
            thresholds = np.unique(training_column)

            # Iterate through each threshold and calculate the information gain
            for threshold in thresholds:
                gain = self.impurity_function(training_labels, training_column, threshold)

                # Update the best gain and split criteria if the current gain is better
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

        
    def information_gain(self, training_labels, training_column, split_threshold):
        """
        Calculate the information gain of a split based on a given threshold.

        Parameters:
        -----------
        training_labels : array-like of shape (n_samples,)
            The target labels corresponding to the training data samples.

        training_column : array-like of shape (n_samples,)
            The values of the feature column used for splitting.

        split_threshold : float
            The threshold value used to split the feature column into two groups.

        Returns:
        --------
        information_gain : float
            The information gain achieved by splitting the data based on the given threshold.

        Notes:
        ------
        This method calculates the information gain of a split by measuring the reduction in entropy
        (impurity) of the parent node compared to the weighted average of entropies of the child nodes
        resulting from the split. It uses the entropy function to compute the entropy of the parent node
        and the child nodes, and then calculates the difference as the information gain.
        """
        # Compute the entropy of the parent node
        parent_entropy = self.entropy(training_labels)

        # Generate left and right indices based on the split threshold
        left_indices, right_indices = self.split(training_column, split_threshold)

        # Check if either child node is empty
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Compute the number of samples in each child node
        n_samples = len(training_labels)
        num_left, num_right = len(left_indices), len(right_indices)

        # Compute the entropy of each child node
        left_entropy = self.entropy(training_labels[left_indices])
        right_entropy = self.entropy(training_labels[right_indices])

        # Compute the weighted average of child entropies
        child_entropy = (num_left / n_samples) * left_entropy + (num_right / n_samples) * right_entropy

        # Calculate the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def gini_index(self, labels):
        """
        Calculate the Gini index for a given set of labels.

        Parameters:
        -----------
        labels : numpy.ndarray of shape (n_samples,)
            The target labels.

        Returns:
        --------
        gini : float
            The Gini index.
        """
        total_samples = len(labels)
        _, label_counts = np.unique(labels, return_counts=True)
        gini = 1.0

        for count in label_counts:
            label_probability = count / total_samples
            gini -= label_probability ** 2

        return gini

    def gini_gain(self, training_labels, training_column, split_threshold):
        """
        Calculate the Gini gain of a split based on a given threshold.

        Parameters:
        -----------
        training_labels : array-like of shape (n_samples,)
            The target labels corresponding to the training data samples.

        training_column : array-like of shape (n_samples,)
            The values of the feature column used for splitting.

        split_threshold : float
            The threshold value used to split the feature column into two groups.

        Returns:
        --------
        gini_gain : float
            The Gini gain achieved by splitting the data based on the given threshold.
        """
        # Compute the Gini index of the parent node
        parent_gini = self.gini_index(training_labels)

        # Generate left and right indices based on the split threshold
        left_indices = [i for i, val in enumerate(training_column) if val <= split_threshold]
        right_indices = [i for i, val in enumerate(training_column) if val > split_threshold]

        # Check if either child node is empty
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Compute the number of samples in each child node
        n_samples = len(training_labels)
        num_left, num_right = len(left_indices), len(right_indices)

        # Compute the Gini index of each child node
        left_gini = self.gini_index(training_labels[left_indices])
        right_gini = self.gini_index(training_labels[right_indices])

        # Compute the weighted average of child Gini indices
        child_gini = (num_left / n_samples) * left_gini + (num_right / n_samples) * right_gini

        # Calculate the Gini gain
        gini_gain = parent_gini - child_gini
        return gini_gain

    def split(self, training_column, split_threshold):
        """
        Split the training data column based on a given threshold.

        Parameters:
        -----------
        training_column : array-like of shape (n_samples,)
            The values of the feature column used for splitting.

        split_threshold : float
            The threshold value used to split the feature column into two groups.

        Returns:
        --------
        left_indices : array-like
            The indices of the samples where the feature values are less than or equal to the split threshold.

        right_indices : array-like
            The indices of the samples where the feature values are greater than the split threshold.

        Notes:
        ------
        This method splits the training data column into two groups based on the specified threshold.
        It returns the indices of samples that belong to the left group (values <= threshold) and the
        indices of samples that belong to the right group (values > threshold).
        """
        # Find indices where feature values are less than or equal to the split threshold
        left_indices = np.argwhere(training_column <= split_threshold).flatten()

        # Find indices where feature values are greater than the split threshold
        right_indices = np.argwhere(training_column > split_threshold).flatten()

        return left_indices, right_indices

    def most_common_label(self, training_labels):
        """
        Determine the most common label in the training data.

        Parameters:
        -----------
        training_labels : array-like of shape (n_samples,)
            The target labels corresponding to the training data samples.

        Returns:
        --------
        most_common : object
            The most common label in the training data.

        Notes:
        ------
        This method counts the occurrences of each label in the training data and returns the label
        that appears most frequently. In the case of ties, it returns the label that appears first.
        """
        # Count the occurrences of each label
        counter = Counter(training_labels)

        # Get the most common label
        most_common = counter.most_common(1)[0][0]

        return most_common

    def predict(self, test_data):
        """
        Predict the target labels for the given test data.

        Parameters:
        -----------
        test_data : array-like of shape (n_samples, n_features)
            The test data samples for which predictions are to be made. Each row represents a sample,
            and each column represents a feature.

        Returns:
        --------
        predictions : array-like of shape (n_samples,)
            The predicted target labels for the test data.

        Notes:
        ------
        This method traverses the decision tree for each sample in the test data and predicts the target label
        based on the learned tree structure. It returns an array of predicted labels corresponding to the test data.
        """
        # Predict target labels for each sample in the test data using tree traversal
        predictions = np.array([self.traverse_tree(sample, self.root) for sample in test_data])

        return predictions

    def entropy(self, class_labels):
        """
        Calculate the entropy of a set of class labels.

        Parameters:
        - class_labels (array-like): A 1D array-like object containing the class labels.

        Returns:
        - entropy_value (float): The entropy value calculated from the input class labels.

        Entropy is a measure of impurity or disorder in a set of class labels. 
        This function calculates the entropy of the input class labels using the following steps:
        
        1. Count occurrences of each unique class label.
        2. Calculate the probability of each class label occurrence.
        3. Compute the entropy using the calculated probabilities.
        
        The entropy is computed using the formula:
        
        entropy_value = -sum(p * log2(p) for p in probabilities if p > 0)
        
        where 'p' represents the probability of each class label.
        
        Note: This function assumes that class labels are represented as integers starting from 0.
        """
            
        # Step 1: Count occurrences of each class label
        label_mapping  = {'very low': 0,'low': 1, 'ok': 2,'good': 3,  'great': 4}
        # Convert each string label to its corresponding integer value using map
        class_labels_int = [label_mapping[label] for label in class_labels]
        class_counts = np.bincount(class_labels_int)
        
        # Step 2: Calculate probability of each class label
        probabilities = class_counts / len(class_labels)
        
        # Step 3: Compute entropy using the calculated probabilities
        entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        
        return entropy_value

    def traverse_tree(self, sample, node):
        """
        Traverse the decision tree to predict the target label for a given sample.

        Parameters:
        -----------
        sample : array-like of shape (n_features,)
            The feature values of the sample for which prediction is to be made.

        node : Node
            The current node being traversed in the decision tree.

        Returns:
        --------
        prediction : object
            The predicted target label for the given sample.

        Notes:
        ------
        This method recursively traverses the decision tree starting from the root node and determines
        the predicted target label for the given sample. It follows the tree's branching rules based on
        feature values until reaching a leaf node, where the predicted label is obtained.
        """
        # Check if the current node is a leaf node
        if node.is_leaf_node():
            # If the current node is a leaf node, return its value as the predicted label
            return node.value

        # If the current node is not a leaf node, traverse further based on the sample's feature value
        if sample[node.feature] <= node.threshold:
            # If the sample's feature value is less than or equal to the node's threshold, traverse left
            return self.traverse_tree(sample, node.left)
        else:
            # If the sample's feature value is greater than the node's threshold, traverse right
            return self.traverse_tree(sample, node.right)

# Utility Functions
def separate_data_and_labels(full_data):
    """
    Separate the data and labels from the full dataset.

    Parameters:
    -----------
    full_data : numpy.ndarray
        The full dataset containing both feature values and target labels.

    Returns:
    --------
    training_data : numpy.ndarray
        The feature values of the dataset, excluding the target labels.

    training_labels : numpy.ndarray
        The target labels corresponding to the dataset.

    Notes:
    ------
    This function separates the feature values and target labels from the full dataset. It assumes
    that the target labels are stored in the first column of the array. Modify the utility_column_index
    variable if the target labels are located elsewhere in the dataset.
    """
    # Find the index of the 'utility' column
    utility_column_index = 0  # Assuming 'utility' is the first column, modify if needed

    # Extract the 'utility' column as training_labels
    training_labels = full_data[:, utility_column_index]

    # Remove the 'utility' column from the ndarray to get the training data
    training_data = np.delete(full_data, utility_column_index, axis=1)

    return training_data, training_labels

def parseDataForDecisionTree(file_path):
    """
    Read data from a file into a numpy DataFrame.
    Note that the data is read as is from the csv file, 
    i.e we are not converting the utilities into ranges (Great, Good, bad, etc)
    in this function
    
    Parameters:
    - file_path (str): Path to the file containing the data.
    
    Returns:
    - df (DataFrame): DataFrame containing the data.
    """
    # Load data from file into numpy array
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
    
    # Create DataFrame
    columns = ['utility', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']
    df = pd.DataFrame(data, columns=columns)
    
    return df

def label_utility_ranges(df, ranges=None):
    """
    Apply labels to the 'utility' column of a DataFrame based on user-specified ranges.
    Note that this does not moddify the original dataframe on which it is called but rather
    returns a new one. This is an intentional design choice because we might want to preserve the
    original real-valued utilities.
    

    Parameters:
    - df (DataFrame): The DataFrame containing the 'utility' column.
    - ranges (dict, optional): User-specified ranges and corresponding labels.
      Keys are tuples representing the lower and upper bounds of the range (inclusive),
      and values are the labels. Default is {(0.0, 0.2): 'very low', (0.2, 0.4): 'low', (0.4, 0.6): 'ok',
     (0.6, 0.8): 'good', (0.8, 1): 'great'}
    
    Returns:
    - new_df (DataFrame): A new DataFrame with the 'utility' column updated with labels.
    """
    # Create a copy of the original DataFrame
    new_df = df.copy()
    
    # Set default ranges and labels if not provided
    if ranges is None:
        ranges = {(0.0, 0.2): 'very low', (0.2, 0.4): 'low', (0.4, 0.6): 'ok',
                (0.6, 0.8): 'good', (0.8, 1): 'great'}
        
    # if ranges is None:
    #     ranges = {(0.0, 0.2): 0, (0.2, 0.4): 1, (0.4, 0.6): 2,
    #             (0.6, 0.8): 3, (0.8, 1): 4}
    
    # Define conditions and corresponding labels based on user-specified or default values
    conditions = [((new_df['utility'] >= lower) & (new_df['utility'] <= upper)) for (lower, upper), label in ranges.items()]
    labels = np.select(conditions, [label for _, label in ranges.items()])
    
    # Update the 'utility' column with labels in the new DataFrame
    new_df['utility'] = labels
    
    return new_df


def accuracy(true_labels, predicted_labels):
    """
    Calculate the accuracy of classification predictions.

    Parameters:
    -----------
    true_labels : array-like of shape (n_samples,)
        The true labels of the samples.

    predicted_labels: array-like of shape (n_samples,)
        The predicted labels of the samples.

    Returns:
    --------
    accuracy : float
        The accuracy of the classification predictions, defined as the proportion of correctly
        classified samples out of the total number of samples.

    Notes:
    ------
    This function computes the accuracy of classification predictions by comparing the true labels
    with the predicted labels and calculating the proportion of correctly classified samples.
    """
    # Count the number of correct predictions and divide by the total number of samples
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_samples = len(true_labels)
    acc = correct_predictions / total_samples

    return acc

def train_test_split(training_data, training_labels, test_size, random_state):
    """
            Split the data into training and testing sets.

            Parameters:
            -----------
            training_data : array-like of shape (n_samples, n_features)
                The training data samples.

            training_labels : array-like of shape (n_samples,)
                The target labels corresponding to the training data samples.

            test_size : float, optional (default=0.2)
                The proportion of the data to include in the test split.

            random_state : int, optional (default=1234)
                Controls the randomness of the split.

            Returns:
            --------
            x_train : array-like
                The training data.

            x_test : array-like
                The testing data.

            y_train : array-like
                The training labels.

            y_test : array-like
                The testing labels.
            """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Number of samples in the training data
    n_samples = len(training_data)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Calculate the number of samples in the test set
    n_test_samples = int(test_size * n_samples)

    # Split the indices into training and test sets
    test_indices = indices[:n_test_samples]
    train_indices = indices[n_test_samples:]

    # Split the data into training and test sets
    x_train, x_test = training_data[train_indices], training_data[test_indices]
    y_train, y_test = training_labels[train_indices], training_labels[test_indices]

    return x_train, x_test, y_train, y_test

class neural_network():
    def __init__(self, learning : float, epochs : int):
        self.epochs = epochs
        self.learning_rate = learning
        self.target_output = []
        self.outputs = []
        self.inputs = []
        self.data = []
        self.num_inputs = 10
        self.weights_input_hidden = []
        self.weights_hidden_output = []

    def activation_function(self, x : float):
        return 1.0 / (1.0 + math.exp(-x))

    def activation_derivative(self, x : float):
        return x * (1.0 - x)

    def error(self):
        sum = 0.0
        for i in range(len(self.target_output)):
            sum += (self.target_output[i] - self.outputs[i]) ** 2

        return (1.0 / (2.0 * len(self.target_output))) * sum

    def parse_data(self):
        self.data = pd.read_csv('Project2_DataSet.txt', delimiter='\t',
                           usecols=['utility', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'])

        # Display the DataFrame

        print(self.data)

    def initialize_neural(self):
        # set up input to hidden weights
        for i in range(self.num_inputs):
            weight = []
            for j in range(self.num_inputs):
                weight.append(0.05)
            self.weights_input_hidden.append(weight)
            
        # set up hidden to output weights
        for i in range(self.num_inputs):
            self.weights_hidden_output.append(0.05)

    def back_propagate(self):
        # for each epoch
        for ignore in range(self.epochs):
            # change range to len(neural.data), but currently takes too long and low is good for testing
            for row in range(3):
                hidden = []
                output = 0.0
            
                # find hidden layer result
                for i in range(self.num_inputs):
                    activation_input = 0.0
                    for x in range(self.num_inputs):
                        activation_input += self.data.iloc[row, x] * self.weights_input_hidden[x][i]
                    
                    hidden.append(self.activation_function(activation_input))
                
                # find output layer result
                out_input = 0.0
                for y in range(self.num_inputs):
                    out_input += hidden[y] * self.weights_hidden_output[y]
                
                output = self.activation_function(out_input)

                delta_output = (self.data.iloc[row, 0] - output) * self.activation_derivative(output)
                
                # update hidden layer weights
                for i in range(self.num_inputs):
                    self.weights_hidden_output[i] += self.learning_rate * delta_output * hidden[i]
                
                # update input layer weights
                delta_hidden = []
                for i in range(self.num_inputs):
                    delta_hidden.append(delta_output * self.weights_hidden_output[i] * self.activation_derivative(hidden[i]))
    
                for i in range(self.num_inputs):
                    for j in range(self.num_inputs):
                        self.weights_input_hidden[j][i] += self.learning_rate * delta_hidden[i] * self.data.iloc[row, j]

    def calc_output(self, row : int):
        hidden = []
            
        for i in range(self.num_inputs):
            activation_input = 0.0
            for x in range(self.num_inputs):
                activation_input += self.data.iloc[row, x] * self.weights_input_hidden[x][i]
                    
            hidden.append(self.activation_function(activation_input))
        
        out_input = 0.0
        for y in range(self.num_inputs):
            out_input += hidden[y] * self.weights_hidden_output[y]
        
        return self.activation_function(out_input)

    def k_fold(self):
        self.initialize_neural()
        # split the data into 5 segments
        split_data = []
        print(len(self.data))
        for i in range(5):
            split_data.append(self.data.iloc[i * int(len(self.data)/5):(i + 1) * int(len(self.data)/5)])

        print(split_data)

        # Run 5 separate learning experiments changing the training and output

        # average results outputted

        print()
        # split
    
    def cost(self, row : int):
        return 0.5 * (self.data.iloc[row, 0] - self.calc_output(row))**2

def main():

    answer = input('1 for Decision Tree, 2 for Back Propagation')

    if answer == '1':
        print('run decision tree')
        DTData = parseDataForDecisionTree('Project2_DataSet.txt')
        full_labeled_data = label_utility_ranges(DTData).values
        training_data, training_labels = separate_data_and_labels(full_labeled_data)

        training_data, testing_data, training_labels, testing_labels = train_test_split(training_data, training_labels, test_size=0.2, random_state=1234)
        
        clf = DecisionTreeClassifier(max_depth=5, impurity_function='gini_gain')
        clf2 = DecisionTreeClassifier(max_depth=5, impurity_function='information_gain')

        clf.fit(training_data, training_labels)
        clf2.fit(training_data, training_labels)

        y_pred = clf.predict(testing_data)
        y2_pred = clf2.predict(testing_data)
        
        acc = accuracy(testing_labels, y_pred)
        acc2 = accuracy(testing_labels, y2_pred)
        print("Accuracy using entropy for impurity: ", acc)
        print("Accuracy using gini index for impurity: ", acc2)

    else:
        neural = neural_network(0.1, 1000)
        neural.parse_data()
        neural.k_fold()
        # change range to len(neural.data), but currently takes too long and low is good for testing
        for i in range(3):
            print(neural.calc_output(i))
        neural.back_propagate()
        print(neural.data['utility'])
        # change range to len(neural.data), but currently takes too long and low is good for testing
        for i in range(3):
            print(neural.calc_output(i))

if __name__ == '__main__':
    main()