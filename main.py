import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from generator import generator
from discriminator import discriminator

print(".................................gpu is possible.................................")
print(torch.cuda.is_available())
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/Random_results'):
    os.mkdir('MNIST_GAN_results/Random_results')
if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
    os.mkdir('MNIST_GAN_results/Fixed_results')


def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix = False):
    z_ = torch.randn((5*5, 100))
    with torch.no_grad():
        z_ = Variable(z_.cuda())

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Epoch')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()



# main function
batch_size = 128
lr = 0.0002
train_epoch = 100

fixed_z_ = torch.randn((5*5, 100)) # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

print(".................................data_loader running.................................")
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

G = generator(input_size=100, n_class = 28*28)
D = discriminator(input_size = 28*28, n_class=1)
G.cuda()
D.cuda()

#loss function
BCE_loss = nn.BCELoss()

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

print(".................................training.................................")
# train
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    for x_, _ in train_loader:
        # train Discriminator D
        D.zero_grad()

        x_ = x_.view(-1, 28*28)

        mini_batch = x_.size()[0]
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        D_result = D(x_)
        #print(D_result.shape)

        D_real_loss = BCE_loss(D_result, y_real_)
        #D_real_score = D_result

        z_ = torch.randn((mini_batch, 100)) # random generate noise picture
        z_ = Variable(z_.cuda())
        G_result = G(z_)
        # print(G_result.shape)

        D_result = D(G_result)
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)) # also random noise
        y_ = torch.ones(mini_batch) # generate label

        z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())
        G_result = G(z_) # G_result is ouputed fake picture
        D_result = D(G_result)
        G_train_loss = BCE_loss(D_result, y_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.item())

    # end for

    print ('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch+1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save = True, path=p, isFix=False)
    show_result((epoch+1), save = True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(D_losses)))

print(".................................saving model running.................................")
# save model
print("Training finish!.. save training results")
torch.save(G.state_dict(), "MNIST_GAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_GAN_results/discriminator_param.pkl")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

# training visible
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')
images = []
for e in range(train_epoch):
    img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)
