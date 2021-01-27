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
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(4 * 4 * 256, hidden_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.conv1.weight.device)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_avg_pool(out)
        out = self.flatten(out)
        out = F.leaky_relu(self.dense(out))
        # out = self.dropout(out)

        return out


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

    def save(self, filename):
        torch.save(self.encoder.state_dict(), filename)

    def load(self, filename):
        self.encoder.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


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
