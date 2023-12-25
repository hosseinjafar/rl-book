import torch.nn as nn
import torch

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

if __name__ == '__main__':
    # example_linear()
    # example_sequntial()
    example_customModule()