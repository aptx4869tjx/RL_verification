class Configurator:
    def __init__(self, initial_state, proposition_list, limited_count, limited_depth, atomic_propositions, formula,
                 get_abstract_state, get_abstract_state_label, get_abstract_state_hash, get_next_states, rtree):
        self.initial_state = initial_state
        self.proposition_list = proposition_list
        self.limited_count = limited_count
        self.limited_depth = limited_depth
        self.atomic_propositions = atomic_propositions
        self.formula = formula
        self.get_abstract_state = get_abstract_state
        self.get_abstract_state_label = get_abstract_state_label
        self.get_abstract_state_hash = get_abstract_state_hash
        self.get_next_states = get_next_states
        self.rtree = rtree
