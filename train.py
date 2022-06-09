from preprocess import Input
import argparse
from model import ConvLSTM
from torch import nn
from torch import optim
import torch
import matplotlib.pyplot as plt
"""
    利用1989年-2020年的海冰数据
    根据2021年过去某一段时间的海冰图像预测后3天的海冰图
"""

parser = argparse.ArgumentParser(description='Sea ice prediction')
parser.add_argument('--epochs', default=500, type=int, help='the train epochs of net work')
parser.add_argument('--lr', default=0.03, type=float, help='the learning rate of the net work')
parser.add_argument('--number_layer', default=64, type=int, help='the number of the GRU/LSTM layer ')
parser.add_argument('--bias', default=True, type=bool, help='whether add bias to the net work')
parser.add_argument('--batch_size', default=32, type=int, help='the batch size of train loader')
args = parser.parse_args()


if __name__ == '__main__':
    input = Input('pic_data', start_year=1989, end_year=2021, predicted_year=2021, days=13,
                  history_days=10, predict_days=1)
    train_loader, test_loader = input.loader()

    net = ConvLSTM(input_channel=3, out_channel=16, hidden_channel=(32, 32), kernel_size=3, num_layers=1,
                   image_size=(446, 302), padding=1)
    net = net.cuda()

    criterion = nn.modules.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(args.epochs):
        loss = torch.zeros(1)
        for i, (data, label) in enumerate(train_loader):
            output = net(data.cuda())
            label = label.squeeze(0)
            label = label.squeeze(0)
            loss = criterion(output.cpu(), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss + loss.item()
        print(f'Epoch {epoch}: {loss.item()}')
        # scheduler.step()

        if (epoch + 1) % 100 == 0:
            for i, (data, label) in enumerate(test_loader):
                output = net(data.cuda())
                output = output.cpu()
                label = label.squeeze(0)
                label = label.squeeze(0)

                loss = criterion(output.cpu(), label)
                print(f'Epoch {epoch}: {loss.item()}')

                output = output.permute(1, 2, 0) * 0.5 + 0.5
                output = output.detach().numpy()

                label = label.permute(1, 2, 0) * 0.5 + 0.5
                label = label.detach().numpy()
                plt.figure()
                plt.imshow(output)
                plt.show()

                plt.figure()
                plt.imshow(label)
                plt.show()
