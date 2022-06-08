class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def remember(self, state, action, reward, new_state, value, log_p):
        # self.input_img.append(input_img)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        self.new_states.append(new_state)
        self.log_probs.append(log_p)
        self.values.append(value)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []
        # self.input_img = []

    def sample_memory(self):
        return self.states, self.actions, self.rewards, self.new_states,\
               self.values, self.log_probs

''' The share memory function is where all the different agents share their memories with each other so that they can
learn from each other's experience'''