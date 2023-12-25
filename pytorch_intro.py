import torch.nn as nn
import torch

import numpy as np
import gym
import cv2

def example_linear():
    l = nn.Linear(2, 5)
    print(l.weight)
    print(l.bias)

    v = torch.Tensor([1, 2]).to(torch.float32)
    print(l(v))

def example_sequntial():
    s = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Dropout(p=0.3),
        nn.Softmax(dim=1)
    )
    print(s)
    input = torch.Tensor([[1, 2]]).to(torch.float32)
    print(s(input))

def example_customModule(training=False):
    class OurModule(nn.Module):
        def __init__(
                self, 
                num_inputs, 
                num_classes, 
                dropout_prob=0.3):
            super(OurModule, self).__init__()
            self.pipe = nn.Sequential(
                nn.Linear(num_inputs, 5),
                nn.ReLU(),
                nn.Linear(5, 20),
                nn.ReLU(),
                nn.Linear(20, num_classes),
                nn.Dropout(p=dropout_prob),
                nn.Softmax(dim=1)
            )
        def forward(self, x):
            return self.pipe(x)
    
    net = OurModule(num_inputs=2, num_classes=3)    
    v = torch.Tensor([[2, 3]]).to(torch.float32)
    out = net(v)
    print(net)
    print(out)

    if training:
        for batch_x, batch_y in iterate_batches(data, brach_size=32):
            batch_x_t = torch.tensor(batch_x)
            batch_y_t = torch.tensor(batch_y)
            out_t = net(batch_x_t)
            loss_t = loss_function(out_t, batch_y_t)
            loss_t.backward()
            optimizer.step()
            optimizer.zero_grad()

def example_GAN():
    class InputWrapper(gym.ObservationWrapper):
        def __init__(self, *args):
            super(InputWrapper, self).__init__(*args)
            assert isinstance(self.observation_space, gym.spaces.Box)
            old_space = self.observation_space
            self.observation_space = gym.spaces.Box(
                self.observation(old_space.low),
                self.observation(old_space.high),
                dtype=np.float32)
        def observation(self, observation):
            new_obs = cv2.resize(
                observation, (IMAGE_SIZE, IMAGE_SIZE)
            )
            # transform (210, 160, 3) -> (3, 210, 160)
            new_obs = np.moveaxis(new_obs, 2, 0)
            return new_obs.astype(np.float32)
    device = "cuda"
    envs = [
        InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
    ]
    input_shape = envs[0].observation_space.shape
    
    net_discr = Discriminator(input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)
    gen_optimizer = torch.optim.Adam(
        net_gener.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(
        params=net_discr.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0
    true_labels_v = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        # fake samples generation. input is 4D: btach, filgers, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, 
            LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)
        

def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)
    while True:
        e = next(env_gen)
        obs, reward, is_done, _, _ = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalizing input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32)
            batch_np *= 2. / 255. - 1.
            yield torch.tensor(batch_np)
            batch.clear()

        if is_done:
            e.reset()



if __name__ == '__main__':
    # example_linear()
    # example_sequntial()
    example_customModule()