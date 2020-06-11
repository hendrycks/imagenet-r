import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # a
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation
        option = np.random.choice(range(9))
        if option == 1:
            # set some weights to zero
            H = ec3.size()[2]
            W = ec3.size()[3]
            mask = (torch.cuda.FloatTensor(H, W).uniform_() > 0.2).float().cuda()
            ec3 = ec3 * mask
            del mask
        elif option == 2:
            # negare some of the weights 
            H = ec3.size()[2]
            W = ec3.size()[3]
            mask = (((torch.cuda.FloatTensor(H, W).uniform_() > 0.1).float() * 2) - 1).cuda()
            ec3 = ec3 * mask
            del mask
        elif option == 3:
            num_channels = 10
            perm = np.array(list(np.random.permutation(num_channels)) + list(range(num_channels, ec3.size()[1])))
            ec3 = ec3[:, perm, :, :]   
        elif option == 4:
            num_channels = ec3.shape[1]
            num_channels_transform = 5
            
            _k = random.randint(1,3)
            _dims = [0, 1, 2]
            random.shuffle(_dims)
            _dims = _dims[:2]

            for i in range(num_channels_transform):
                filter_select = random.choice(list(range(num_channels)))
                ec3[:,filter_select] = torch.flip(ec3[:,filter_select], dims=_dims)
        elif option == 5:
            num_channels = ec3.shape[1]
            num_channels_transform = num_channels
            
            _k = random.randint(1,3)
            _dims = [0, 1, 2]
            random.shuffle(_dims)
            _dims = _dims[:2]

            for i in range(num_channels_transform):
                if i == num_channels_transform / 2:
                    _dims = [_dims[1], _dims[0]]
                ec3[:,i] = torch.flip(ec3[:,i], dims=_dims)
        elif option == 6:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                for j in range(z.size(0)):
                    shift_x, shift_y = 1, 1# np.random.randint(3, size=(2,))
                    z[j,j,shift_x,shift_y] = 1 # np.random.choice([1.,-1.])
                
                # Without this line, z would be the identity convolution
                z = z + ((torch.rand_like(z) - 0.5) * 0.2)
                ec3 = F.conv2d(ec3, z, padding=1)
                del z
        elif option == 7:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                for j in range(z.size(0)):
                    shift_x, shift_y = 1, 1# np.random.randint(3, size=(2,))
                    z[j,j,shift_x,shift_y] = 1 # np.random.choice([1.,-1.])
                    
                    if random.random() < 0.5:
                        rand_layer = random.randint(0, c - 1)
                        z[j, rand_layer, random.randint(-1, 1), random.randint(-1, 1)] = 1
                
                ec3 = F.conv2d(ec3, z, padding=1)
                del z
        elif option == 8:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                shift_x, shift_y = np.random.randint(3, size=(2,))
                for j in range(z.size(0)):
                    if random.random() < 0.2:
                        shift_x, shift_y = np.random.randint(3, size=(2,))

                    z[j,j,shift_x,shift_y] = 1 # np.random.choice([1.,-1.])

                # Without this line, z would be the identity convolution
                # z = z + ((torch.rand_like(z) - 0.5) * 0.2)
                ec3 = F.conv2d(ec3, z, padding=1)
                del z

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
        if option == 0: 
            self.encoded = self.encoded *\
            (3 + 2 * np.float32(np.random.uniform()) * (2*torch.rand_like(self.encoded-1)))
        return self.decode(self.encoded)

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
