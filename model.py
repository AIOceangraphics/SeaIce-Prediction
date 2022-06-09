import torch
from torch import nn
import math
from torch.nn import init
from torchvision.models import resnet34


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, hidden_channel, out_channel, num_layers, kernel_size, image_size, padding, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.padding = padding
        self.bias = bias

        self.conv_x = nn.Conv2d(input_channel, 4*self.hidden_channel, kernel_size=self.kernel_size,
                                padding=self.padding, bias=True)
        self.conv_h = nn.Conv2d(hidden_channel, 4*self.hidden_channel, kernel_size=self.kernel_size, padding=self.padding)

        self.w_cc = nn.Parameter(torch.empty(3*hidden_channel, image_size[0], image_size[1]))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_channel)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h, c):
        x = self.conv_x(x)
        h = self.conv_h(h)
        xi, xf, xc, xo = torch.split(x, self.hidden_channel, dim=1)
        hi, hf, hc, ho = torch.split(h, self.hidden_channel, dim=1)
        ci, cf, co = torch.split(self.w_cc, self.hidden_channel, dim=0)
        
        i = torch.sigmoid(xi + hi + ci * c)
        f = torch.sigmoid(xf + hf + cf * c)
        c = f * c + i * torch.tanh(xc + hc)
        o = torch.sigmoid(xo + ho + co)
        h = o * torch.tanh(c)
        
        return h, c

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_channel, height, width).cuda()
        c = torch.zeros(batch_size, self.hidden_channel, height, width).cuda()

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channel, out_channel, hidden_channel, kernel_size, num_layers, image_size, padding, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.out_channel = out_channel
        self.image_size = image_size
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

        cell_list = []
        for i in range(self.num_layers):
            temp_input_channel = self.input_channel if i == 0 else self.hidden_channel[i - 1]
            cell_list.append(ConvLSTMCell(temp_input_channel, self.hidden_channel[i], kernel_size=self.kernel_size,
                                          bias=self.bias, out_channel=64, num_layers=num_layers, image_size=image_size,
                                          padding=self.padding))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x):
        b, _, _, h, w = x.size()
        hidden_cell = self.init_hidden(b, (h, w))

        layer_output_list = []
        last_state = []
        day_len = x.size(1)
        temp_input = x

        for i in range(self.num_layers):
            temp_hidden_cell = hidden_cell[i]
            h = temp_hidden_cell[0]
            c = temp_hidden_cell[1]
            output = []
            for j in range(day_len):
                h, c = self.cell_list[i](temp_input[:, j, :, :, :], h, c)
                output.append(h)  # 这层的hidden值

            layer_output = torch.stack(output, dim=1)
            temp_input = layer_output
            layer_output_list.append(layer_output)
            last_state.append([h, c])

        # 将特征组合
        hidden_state = torch.cat(last_state[-1], dim=1)
        output = self.conv(hidden_state)
        output = output.squeeze(0)
        return output

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


# class IceNet(nn.Module):
#     def __init__(self, input_channel, hidden_channel, kernel_size, num_layers):
#         super(IceNet, self).__init__()
#         self.cnn = resnet34(pretrained=False)
#         self.rnn = ConvLSTM(input_channel, hidden_channel, kernel_size, num_layers)
#
#     def forward(self, x):
#
