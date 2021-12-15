from Generator import *
from Discriminator import *
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import torch.autograd
from torchvision import datasets
import torch.utils.data as Data
from torch.autograd import Variable

discriminator=Discriminator()
generator=Generator()

d_optimizer=optim.Adam(discriminator.parameters(),lr=2e-4)

g_optimizer=optim.Adam(generator.parameters(),lr=2e-4)

loss=nn.BCELoss()

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


def real_data_target(size):
    data=torch.ones(size,1)
    return data

def fake_data_target(size):
    data=torch.zeros(size,1)
    return data

def noise(size):
    n=torch.randn(size,100)
    return n


if __name__ == '__main__':
    num_epoch=1000
    z_dim=100
    batch_size=128
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1,), (0.5,))
    ])
    mnist=datasets.MNIST(root='./mnist/',train=True, transform=img_transform,download=False)

    dataloader=Data.DataLoader(
        dataset=mnist,
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(num_epoch):
        for i,(img,_) in enumerate(dataloader):
            num_img=img.size(0)

            img = img.view(num_img,-1)
            real_img=Variable(img)
            real_label=Variable(real_data_target(num_img))
            fake_label=Variable(fake_data_target((num_img)))
            real_out=discriminator(real_img)
            d_loss_real=loss(real_out,real_label)
            real_scores=real_out
            z=Variable(torch.randn(num_img,z_dim))
            fake_img=generator(z)
            fake_out=discriminator(fake_img)
            d_loss_fake=loss(fake_out , fake_label)
            fake_scores=fake_out
            d_loss=d_loss_real+d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # ==================训练生成器============================
            ################################生成网络的训练###############################
            # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
            # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
            # 反向传播更新的参数是生成网络里面的参数，
            # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
            # 这样就达到了对抗的目的
            # 计算假的图片的损失
            z = Variable(torch.randn(num_img, z_dim))  # 得到随机噪声
            fake_img = generator(z)  # 随机噪声输入到生成器中，得到一副假的图片
            #
            output = discriminator(fake_img)  # 经过判别器得到的结果
            g_loss = loss(output, real_label)  # 得到的假的图片与真实的图片的label的loss
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
            # 打印中间的损失
            if (i + 1) % 100 == 0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                      'D real: {:.6f},D fake: {:.6f}'.format(
                    epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                    real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
                ))
            if epoch == 0 and i == len(dataloader) - 1:
                real_images = to_img(real_img.data)
                save_image(real_images, './img2/real_images.png')
            if i == len(dataloader) - 1:
                fake_images = to_img(fake_img.data)
                save_image(fake_images, './img2/fake_images-{}.png'.format(epoch + 1))
        # 保存模型
        torch.save(generator.state_dict(), './generator.pth')
        torch.save(discriminator.state_dict(), './discriminator.pth')



