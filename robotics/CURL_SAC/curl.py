import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

from robotics.CURL_SAC.utils import Encoder


class CURL:
    def __init__(self, hidden_dim, neg_n=8, tau=1e-2, device='cuda'):
        self.hidden_dim = hidden_dim
        self.neg_n = neg_n
        self.tau = tau
        self.device = device

        self.W = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, requires_grad=True, device=device))

        self.encoder = Encoder(hidden_dim).to(device)
        self.momentum_encoder = deepcopy(self.encoder).to(device)

        self.opt = torch.optim.Adam(list(self.encoder.parameters()) + [self.W], lr=1e-4)

    def __call__(self, x):
        out = self.encoder(x)

        return out

    def train(self, obs, batch_size=32, epochs=10):
        # obs should be 4-d tensor with (batch_size, 3*num_frames, W, H)
        # where num_frames is number stacked frames for speed capture.

        # in CURL_SAC query and keys are processed to get contrastive loss
        # for each ob in obs random crop is made and self.neg_n random crops
        # from different images are made also.
        for epoch in range(epochs):
            idxs = np.random.randint(0, obs.shape[0], size=batch_size)
            batch_obs = obs[idxs, ...].to(self.device)
            query_crops = torch.stack([self.random_crop(ob) for ob in batch_obs], dim=0)
            key_crops = torch.stack([self.random_crop(ob) for ob in batch_obs], dim=0)
            query_enc = self.encoder(query_crops)
            key_enc = self.momentum_encoder(key_crops).detach()
            Wz = torch.matmul(self.W, key_enc.T)
            logits = torch.matmul(query_enc, Wz)
            # subtract max from logits for stability according to original paper
            logits = logits - torch.max(logits, dim=1)[0]
            labels = torch.arange(logits.shape[0]).to(self.device)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        # for i in range(len(obs)):
        #     ob = torch.FloatTensor(obs[i])[np.newaxis, ...].to(self.device)
        #     query = self.encoder(self.random_crop(ob))
        #     positive = self.momentum_encoder(self.random_crop(ob))
        #     keys = []
        #     for _ in range(self.neg_n):
        #         idx = i
        #         while idx == i:  # ensure idx doesn't equal to i
        #             idx = np.random.randint(0, len(obs))
        #         fake_ob = torch.FloatTensor(obs[idx]).to(self.device)[np.newaxis, ...]
        #         key = self.momentum_encoder(self.random_crop(fake_ob)).detach()
        #         keys.append(key)
        #
        #     numerator = torch.exp(torch.matmul(query, torch.matmul(self.W, positive.T)))
        #     denominator = torch.exp(torch.matmul(query, torch.matmul(self.W, positive.T)) +
        #                             sum([torch.matmul(query, torch.matmul(self.W, k.T)) for k in keys]))
        #
        #     loss = torch.log(numerator / denominator)
        #     loss.backward()
        #     self.opt.step()
        #     self.opt.zero_grad()

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
    states = [torch.FloatTensor(state)]

    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        states.append(torch.FloatTensor(next_state))
        if done:
            break

    print("Training")
    curl.train(torch.stack(states))
