import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

'''In the inverse model you want to predict the action the agent took to cause this state to transition from time t to t+1
So you are comparing an integer vs an actual label/ the actual action the agent took
Multi-class classification problem
This is a cross entropy loss between the predicted action and the actual action the agent took'''
"The loss for the forward model is the mse between the predicted state at time t+1 and the actua state at time t+1  "
"So we have two losses : one that comes from the inverse model and one that comes from the forward model "


class Encoder(nn.Module):

    def __init__(self, input_dims, feature_dim=288):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

    def forward(self, img):
        enc = F.elu(self.conv1(img))
        enc = F.elu(self.conv2(enc))
        enc = F.elu(self.conv3(enc))
        enc = F.elu(self.conv4(enc))

        enc_flatten = T.flatten(enc, start_dim=1)
        # enc_flatten = enc.view((enc.size()[0], -1))
        # print(enc_flatten.shape, 'flatten')

        return enc_flatten

    # OR
    '''def __init__(self, input_dims, feature_dim=288):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(input_dims[0], 32, (3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)

        shape = self.get_shape(input_dims)
        # Layer that will extract the features
        self.fc1 = nn.Linear(shape, feature_dim)

    def get_shape(self, input_dims):
        img = T.zeros(1, *input_dims)
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        shape = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        # return int(np.prod(x.size()))
        return shape

    def forward(self, img):
        enc = F.elu(self.conv1(img))
        enc = F.elu(self.conv2(enc))
        enc = F.elu(self.conv3(enc))
        enc = self.conv4(enc)
        # [T, 32, 3, 3] to [T, 288]
        enc_flatten = T.flatten(enc, start_dim=1)
        # conv = enc.view(enc.size()[0], -1).to(T.float)
        features = self.fc1(enc_flatten)

        return features'''


class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=4, alpha=0.1, beta=0.2, feature_dims=288):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # self.encoder = Encoder(input_dims)

        self.l4_encoder = Encoder(input_dims)
        self.l5_encoder = Encoder(input_dims)
        # inverse model:Given a succession of states what actions was taken
        self.inverse = nn.Linear(feature_dims * 2, 256)
        # Give us the logits of our policy.
        # We are going to pass this over the cross entropy function
        # so were not going to be doing any soft max activation
        self.pi_logits = nn.Linear(256, n_actions)
        # Forward Model
        # Given a state and action what is going to be the next state/representation
        # (takes the feature representations plus the action)
        self.dense1 = nn.Linear(feature_dims + 1, 256)
        # Resulting state representation/ predicted new_state
        self.predicted_new_state = nn.Linear(256, feature_dims)

        device = T.device('cpu')
        self.to(device)

    ''' The prediction module takes in a state St 
    and action at and produces a prediction for the subsequent state S t+1 '''

    # Forward model takes the action and the current state and predicts the next state
    def forward(self, obs, new_obs, action):
        # Pass the state and new_state through our convolutional layer to get the features representations
        '''state = self.encoder(obs)
        with T.no_grad():
            new_state = self.encoder(new_obs)'''
            # new_state = self.l5_encoder(new_obs)
        state = self.l4_encoder(obs)
        new_state = self.l5_encoder(new_obs)

        # for encoding random features
        '''with T.no_grad():
            state = self.l4_encoder(obs)
            new_state = self.l5_encoder(new_obs)'''
        # new_state = self.l5_encoder(new_obs)

        state = state.to(T.float)
        new_state = new_state.to(T.float)

        # concatenate state(features at time step t) and new_state(features at time step t+1)
        inverse = self.inverse(T.cat([state, new_state], dim=1))
        pi_logits = self.pi_logits(inverse)

        # Forward Operation reshape action from [T] to [T, 1]
        action = action.reshape((action.size()[0], 1))
        # concatenate state and action to get the predicted state phi_hat_new
        forward_input = T.cat([state, action], dim=1)
        dense = self.dense1(forward_input)
        predicted_new_state = self.predicted_new_state(dense)

        return new_state, pi_logits, predicted_new_state

    '''def save_models(self, input_dims):
        # self.actor_critic.save(self.checkpoint_file)
        np.save(os.path.join('./', 'icm'), ICM(input_dims))
        print('... saving models ...')'''

    '''This prediction along with the true next state are passed to a mean-squared error (or some other error) function 
    which produces the prediction error'''

    def calc_loss(self, states, new_states, actions):
        # don't need [] b/c these are lists of states
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        new_states = T.tensor(new_states, dtype=T.float)

        # Get the state, new state and pass it through our forward operation function
        new_state, pi_logits, predicted_new_state = self.forward(states, new_states, actions)
        # print(actions)
        # print(pi_logits)

        inverse_loss = nn.CrossEntropyLoss()
        L_I = (1 - self.beta) * inverse_loss(pi_logits, actions.to(T.long))

        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(predicted_new_state, new_state)

        intrinsic_reward = self.alpha * 0.5 * ((predicted_new_state - new_state).pow(2)).mean(dim=1)
        return intrinsic_reward, L_I, L_F
