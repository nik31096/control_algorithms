import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy


class BasicBlock(nn.Module):
    def __init__(self, in_maps, out_maps, downsample=False):
        super(BasicBlock, self).__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.conv1 = nn.Conv2d(in_maps, out_maps, (3, 3), stride=1 if not downsample else 2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_maps)
        self.conv2 = nn.Conv2d(out_maps, out_maps, (3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_maps)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_maps, out_maps, (1, 1), stride=2),
                nn.BatchNorm2d(out_maps)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        out = F.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (7, 7), stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d((3, 3), stride=1, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, downsample=True),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(8 * 8 * 128, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.dense2 = nn.Linear(512, hidden_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = F.relu(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)

        return out


class CURL:
    def __init__(self, hidden_dim, neg_n=8, tau=1e-2):
        self.hidden_dim = hidden_dim
        self.neg_n = neg_n
        self.tau = tau

        self.W = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, requires_grad=True))

        self.encoder = Encoder(hidden_dim)
        self.momentum_encoder = deepcopy(self.encoder)

        self.opt = torch.optim.Adam(list(self.encoder.parameters()) + [self.W], lr=1e-4)

    def __call__(self, x):
        out = self.encoder(x)

        return out

    def train(self, obs):
        # obs should be 4-d tensor with (batch_size, 3*num_frames, W, H)
        # where num_frames is number stacked frames for speed capture.

        # in CURL query and keys are processed to get contrastive loss
        # for each ob in obs random crop is made and self.neg_n random crops
        # from different images are made also.

        for i in range(len(obs)):
            ob = torch.FloatTensor(obs[i])[np.newaxis, ...]
            query = self.encoder(self.random_crop(ob))
            positive = self.momentum_encoder(self.random_crop(ob))
            keys = []
            for _ in range(self.neg_n):
                idx = i
                while idx == i: # ensure idx doesn't equal to i
                    idx = np.random.randint(0, len(obs))
                key = self.momentum_encoder(self.random_crop(torch.FloatTensor(obs[idx])[np.newaxis, ...])).detach()
                keys.append(key)


            numerator = torch.exp(torch.matmul(query, torch.matmul(self.W, positive.T)))
            denominator = torch.exp(torch.matmul(query, torch.matmul(self.W, positive.T)) + \
                                    sum([torch.matmul(query, torch.matmul(self.W, k.T)) for k in keys]))

            loss = torch.log(numerator / denominator)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        self._soft_update()

    def _soft_update(self):
        for params, target_params in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            target_params.data.copy_((1 - self.tau)*target_params + self.tau*params)

    @staticmethod
    def random_crop(img, size=(64, 64)):
        # suppose img is in CHW format
        left_top_corner = (np.random.randint(0, img.shape[-2] - size[0]), np.random.randint(0, img.shape[-1] - size[1]))
        crop = img[..., left_top_corner[0]:left_top_corner[0] + size[0], left_top_corner[1]:left_top_corner[1] + size[1]]

        return crop


if __name__ == '__main__':
    import dmc2gym

    curl = CURL(10)

    env = dmc2gym.make("cheetah", "run", from_pixels=True, visualize_reward=False)
    state = env.reset()
    states = [state]

    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        states.append(next_state)
        if done:
            break

    curl.train(states)
