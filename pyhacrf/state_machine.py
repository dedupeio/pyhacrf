import numpy as np
from collections import defaultdict, deque


class GeneralStateMachine(object):
    """ State machine which, together with two input sequences, is used to build the lattice.

    Each state and each transition is labelled by different integers.

    Parameters
    ----------
    start_states : list of ints
        The states that the state machine can start in.

    transitions : List of tuples
        The start state, end state, and number of positions to move in each sequence. For example,
        [(0, 0, (0, 1)),  # insertion into the first sequence, while going from state 0 to state 0.
         (1, 0, (1, 0)),  # deletion from first sequence, while moving from state 1 to state 0.
         (2, 1, (1, 1)),  # match/substitution - move from state 2 to state 1.
         ...
        ]

    states_to_classes : dictionary
        Dictionary where each state is mapped to a class.
    """

    def __init__(self, start_states, transitions, states_to_classes):
        self._start_states = start_states
        self._transitions = transitions

        max_state = max(max(s for s, _, _ in transitions), max(s for _, s, _ in transitions)) + 1
        self.n_states = max_state
        self.n_transitions = len(transitions)
        self.states_to_classes = states_to_classes

    def build_lattice(self, x):
        """ Construct the list of nodes and edges for input features. """
        I, J, _ = x.shape
        start_states, transitions = self._start_states, self._transitions

        lattice = []
        transitions_d = defaultdict(list)
        for transition_index, (s0, s1, delta) in enumerate(transitions):
            transitions_d[s0].append((s1, delta, transition_index))
        # Add start states
        unvisited_nodes = deque([(0, 0, s) for s in start_states])
        visited_nodes = set()
        n_states = self.n_states

        while unvisited_nodes:
            node = unvisited_nodes.popleft()
            lattice.append(node)
            i, j, s0 = node
            for s1, delta, transition_index in transitions_d[s0]:
                try:
                    di, dj = delta
                except TypeError:
                    di, dj = delta(i, j, x)

                if i + di < I and j + dj < J:
                    edge = (i, j, s0, i + di, j + dj, s1, transition_index + n_states)
                    lattice.append(edge)
                    dest_node = (i + di, j + dj, s1)
                    if dest_node not in visited_nodes:
                        unvisited_nodes.append(dest_node)
                        visited_nodes.add(dest_node)

        lattice.sort()

        # Step backwards through lattice and add visitable nodes to the set of nodes to keep. The rest are discarded.
        final_lattice = []
        visited_nodes = set((I-1, J-1, s) for s in range(n_states))

        for node in lattice[::-1]:
            if node in visited_nodes:
                final_lattice.append(node)
            elif len(node) > 3:
                source_node, dest_node = node[0:3], node[3:6]
                if dest_node in visited_nodes:
                    visited_nodes.add(source_node)
                    final_lattice.append(node)

        reversed_list = list(reversed(final_lattice))

        # Squash list
        lattice = [edge for edge in reversed_list if len(edge) > 3]
        return np.array(lattice, dtype='int64')

class DefaultStateMachine(object):
    """ State machine which, together with two input sequences, is used to build the lattice.

    Simple and fast state machine with a single state for each class.
    Allows for character match/substitution, deletion, and insertion.

    Parameters
    ----------
    classes : list
        The set of labels.
    """
    def __init__(self, classes) :
        self.n_states = len(classes)
        self.classes = classes
        self.states_to_classes = {i: c
                                  for i, c
                                  in enumerate(classes)}
        self.n_transitions = 3 * len(classes)
        
