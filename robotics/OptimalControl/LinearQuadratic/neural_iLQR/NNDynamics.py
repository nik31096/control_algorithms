import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import namedtuple
import cloudpickle


class LinearDynamicsNet(nn.Module):
    def __init__(self, state_dim, action_dim, stochastic=False):
        super(LinearDynamicsNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, 128), nn.ReLU())
        self.common_layer = nn.Linear(256, 128)
        self.matrixA = nn.Linear(128, state_dim*state_dim)
        self.matrixB = nn.Linear(128, state_dim*action_dim)

    def obtain_system(self, state, action):
        processed_state = self.state_encoder(state)
        processed_action = self.action_encoder(action)
        out = torch.cat([processed_state, processed_action], dim=-1)
        out = F.relu(self.common_layer(out))
        A = self.matrixA(out)
        A = torch.reshape(A, (self.state_dim, self.state_dim))

        B = self.matrixB(out)
        B = torch.reshape(B, (self.state_dim, self.action_dim))

        return A, B

    def forward(self, state, action):
        A, B = self.obtain_system(state, action)
        next_state = torch.dot(A, state) + torch.dot(B, action)
        if self.stochastic:
            normal = torch.distributions.Normal(loc=next_state, scale=1)
            next_state = normal.sample()

        return next_state


class LinearDynamics:
    def __init__(self, state_dim, action_dim, device='cuda', derivative_method='nn', stochastic=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.derivative_method = derivative_method
        self.stochastic = stochastic

        self.buf = ExperienceReplay(device=device)

        self.build_network()

    def build_network(self):
        self.net = LinearDynamicsNet(self.state_dim, self.action_dim, stochastic=self.stochastic)

        self.opt = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.opt.zero_grad()

    def train_dynamics(self, batch_size=32, epochs=10):
        losses = []
        for epoch in range(epochs):
            states, actions, next_states = self.buf.sample(batch_size)
            pred_next_states = self.net(states, actions)
            loss = torch.mean((next_states - pred_next_states)**2)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            losses.append(loss.cpu().data.numpy())

        return losses

    def add_trajectory(self, states, actions, next_states):
        self.buf.put(states, actions, next_states)

    def dyn(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = self.net(state, action)

        return next_state.cpu().data.numpy()

    def derivative(self, arg, h=1e-5, method=None):
        def get_derivative(state, action):
            A, B = self.net.obtain_system(state, action)
            if arg == 0:
                return A
            elif arg == 1:
                return B

        return get_derivative


class NonLinearDynamicsNet(nn.Module):
    def __init__(self, state_dim, action_dim, device='cuda'):
        super(NonLinearDynamicsNet, self).__init__()
        self.device = device

        self.state_layer = nn.Linear(state_dim, 256)
        self.action_layer = nn.Linear(action_dim, 256)

        self.hid_layer = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, state_dim)

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)

        proc_state = F.leaky_relu(self.state_layer(state))
        proc_action = F.leaky_relu(self.action_layer(action))
        out = torch.cat([proc_state, proc_action], dim=-1)
        out = torch.tanh(self.hid_layer(out))
        next_state = self.output_layer(out)

        return next_state


class NonLinearDynamics(LinearDynamics):
    def __init__(self, state_dim, action_dim, device='cuda', derivative_method='nn', stochastic=False):
        if not derivative_method in ['nn', 'batch', 'forward', 'backward', 'center']:
            raise ValueError("Unknown value of derivative method!")

        super(NonLinearDynamics, self).__init__(state_dim, action_dim, device, derivative_method, stochastic)

    def build_network(self):
        self.net = NonLinearDynamicsNet(self.state_dim, self.action_dim, device=self.device).to(self.device)

        self.opt = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.opt.zero_grad()

    def train_dynamics(self, batch_size=128, epochs=50):
        losses = []
        for epoch in range(epochs):
            states, actions, next_states = self.buf.sample(batch_size=batch_size)
            predicted_states = self.net(states, actions)
            loss = torch.mean((predicted_states - next_states)**2)

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            losses.append(loss.cpu().data.numpy())

        return losses

    def derivative(self, arg, h=1e-5, method=None):
        if method is None:
            method = self.derivative_method

        def get_derivative(state, action):
            assert arg in [0, 1], "Dynamics function can have only two arguments: state and action."
            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            base = self.net(state, action)
            if arg == 0:
                if method == 'nn':
                    func = lambda s: self.net(s, action)
                    jacobian = torch.autograd.functional.jacobian(func=func, inputs=state, create_graph=False)

                elif method == 'batch':
                    extended_state = state.repeat(state.shape[0], 1)
                    extended_action = action.repeat(state.shape[0], 1)
                    perturbed_state = extended_state + h*torch.eye(state.shape[0])
                    jacobian = (self.net(perturbed_state, extended_action) -
                                self.net(extended_state, extended_action)) / h
                    jacobian = jacobian.T

                elif method == 'forward':
                    d = []
                    for i in range(self.state_dim):
                        o = torch.zeros((self.state_dim,))
                        o[i] = 1
                        perturbed_state = state + h*o
                        j = (self.net(perturbed_state, action) - base) / h
                        d.append(j)

                    jacobian = torch.stack(d, dim=1)

                elif method == 'backward':
                    d = []
                    for i in range(self.state_dim):
                        o = torch.zeros((self.state_dim,))
                        o[i] = 1
                        perturbed_state = state - h * o
                        j = (base - self.net(perturbed_state, action)) / h
                        d.append(j)

                    jacobian = torch.stack(d, dim=1)

                elif method == 'center':
                    d = []
                    for i in range(self.state_dim):
                        o = torch.zeros((self.state_dim,))
                        o[i] = 1
                        forward_state = state + h * o / 2
                        backward_state = state - h * o / 2
                        j = (self.net(forward_state, action) - self.net(backward_state, action)) / h
                        d.append(j)

                    jacobian = torch.stack(d, dim=1)
            else:
                if method == 'nn':
                    func = lambda a: self.net(state, a)
                    jacobian = torch.autograd.functional.jacobian(func=func, inputs=action, create_graph=False)

                elif method == 'batch':
                    extended_state = state.repeat(state.shape[0], 1)
                    extended_action = action.repeat(state.shape[0], 1)
                    perturbed_action = extended_state + h * torch.eye(state.shape[0])
                    jacobian = (self.net(extended_state, perturbed_action) -
                                self.net(extended_state, extended_action)) / h
                    jacobian = jacobian.T

                elif method == 'forward':
                    d = []
                    for i in range(self.action_dim):
                        o = torch.zeros((self.action_dim,))
                        o[i] = 1
                        perturbed_action = action + h * o
                        j = (self.net(state, perturbed_action) - base) / h
                        d.append(j)

                    jacobian = torch.stack(d, dim=1)

                elif method == 'backward':
                    d = []
                    for i in range(self.action_dim):
                        o = torch.zeros((self.action_dim,))
                        o[i] = 1
                        perturbed_action = action - h * o
                        j = (base - self.net(state, perturbed_action)) / h
                        d.append(j)

                    jacobian = torch.stack(d, dim=1)

                elif method == 'center':
                    d = []
                    for i in range(self.action_dim):
                        o = torch.zeros((self.action_dim,))
                        o[i] = 1
                        forward_action = action + h * o / 2
                        backward_action = action - h * o / 2
                        j = (self.net(state, forward_action) - self.net(state, backward_action)) / h
                        d.append(j)

                    jacobian = torch.stack(d, dim=1)

            return jacobian.cpu().data.numpy()

        return get_derivative

    def save(self, name):
        torch.save(self.net.state_dict(), name)

    def load(self, name):
        self.net.load_state_dict(torch.load(name, map_location=lambda storage, location: storage))


class ExpRepItem:
    def __init__(self, state, action, next_state):
        self.state, self.action, self.next_state = state, action, next_state


class ExperienceReplay:
    def __init__(self, size=100000, device='cuda'):
        self.size = size
        self.data = []
        self._next = 0
        self.device = device

    def put(self, states, actions, next_states):
        for i in range(len(states)):
            state = torch.FloatTensor(states[i])
            action = torch.FloatTensor(actions[i])
            next_state = torch.FloatTensor(next_states[i])
            transition = self._get_transition(state, action, next_state)
            if self._next >= len(self.data):
                self.data.append(transition)
            else:
                self.data[self._next] = transition

            self._next = (self._next + 1) % self.size

    @staticmethod
    def _get_transition(state, action, next_state):
        return ExpRepItem(state, action, next_state)

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size):
        states, actions, next_states = [], [], []
        idxs = np.random.choice(len(self.data), batch_size, replace=False)
        for idx in idxs:
            sample = self.data[idx]
            states.append(sample.state)
            actions.append(sample.action)
            next_states.append(sample.next_state)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        return states, actions, next_states
