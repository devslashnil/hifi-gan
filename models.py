import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    # инитиализация блока быстрого доступа
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        # для создаваемых моделей вызывем эту строчку в конструкторе
        super(ResBlock1, self).__init__()
        # сохраняем гиперпараметры
        self.h = h
        # создаем список 3ых первых подряд идущих расширенные свёртки с расширениями (1, 3, 5)
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])

        # инитиализирует веса mean=0 std=0.01
        self.convs1.apply(init_weights)

        # создаем список 3ых подряд идущих свёрток с расширением 1
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        # инитиализирует веса mean=0 std=0.01
        self.convs2.apply(init_weights)

    # Функция forward вычисляет результирующие тензоры из входных тензоров.
    def forward(self, x):
        # проходим одновременно по двум спискам с расширенными свертками
        for c1, c2 in zip(self.convs1, self.convs2):
            # пропускаем вход через ReLU, с а = 0.1
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # пропускаем вход через расширенную свертку с заданным из гипперпараметров расширением
            xt = c1(xt)
            # пропускаем вход через ReLU, с а = 0.1
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # пропускаем вход через расширенную свертку с расширением 1
            xt = c2(xt)
            # между каждой парой слоёв добавляем исходный входной вектор к результату от слоя
            # за счет этого шага и будет происходить процесс пропуска слоев
            x = xt + x
        return x

    # TODO:
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    # инитиализация генератора
    def __init__(self, h):
        # для создаваемых моделей вызывем эту строчку в конструкторе
        super(Generator, self).__init__()
        # создаем атрибут с переданными гипперпараметрами
        self.h = h
        # задаем кол-во ядер для блоков с быстрым доступом (ядра [3,7,11])
        self.num_kernels = len(h.resblock_kernel_sizes)
        # задаем кол-во слоев повышения дискритизации (множители [8,8,2,2])
        self.num_upsamples = len(h.upsample_rates)
        # создаем входной слой (входных канало 80, результирующих 512, ядро 7x1, шаг 1, отступ 3
        # weight_norm - проводит нормализацию весов
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        # выбор между типами блоков быстрого доступа, в работе используется только первый тип
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        # создаем список для хранения апсемплинговых моделей
        self.ups = nn.ModuleList()
        # одновременно итерируем по значеням множетелей апсемплинговых слоёв [8,8,2,2] и размерам их ядер [16,16,4,4]
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # добавляем слои транспонированных свёрток
            # сужаем кол-во каналов на два в степени номера итерация
            # падинг задаем как разницу множителя апсемплинга на размер ядра
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        # создаем список для хранения блоков быстрого доступа
        self.resblocks = nn.ModuleList()
        # каждому апсемплиноговому слою будет соответсовать последующий слой быстрого доступа
        for i in range(len(self.ups)):
            # снижаем кол-во каналов на 2 в степени итерация
            ch = h.upsample_initial_channel//(2**(i+1))
            # одновременно итерируем по размерам ядра [3,7,11] и размерам расширений [1,3,5], [1,3,5], [1,3,5]]
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                # добавляем слои блоков быстрого доступа
                self.resblocks.append(resblock(h, ch, k, d))

        # создаем результирующий слой, сужающий все каналы в 1
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        # для каждого элемента списка слоев проводим инитилизацию весов
        # init_weights задает нормальное распредение для весов, где среднее 0, а стандартное отклонение 0.01
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    # Функция forward вычисляет результирующие тензоры из входных тензоров.
    def forward(self, x):
        # пропускаем вход через входной слой
        x = self.conv_pre(x)
        # проходим по парам слой апсемплинга/слой быстрого доступа
        for i in range(self.num_upsamples):
            # пропускаем вход через ReLU, с а = 0.1
            x = F.leaky_relu(x, LRELU_SLOPE)
            # пропускаем вход через i-тый апсемлинговый слой
            x = self.ups[i](x)
            # TODO: after res
            xs = None
            # проходим бо всем ядрам слоев быстрого доступа
            for j in range(self.num_kernels):
                if xs is None:
                    # TODO: after res
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    # TODO: after res
                    xs += self.resblocks[i*self.num_kernels+j](x)
            # так как xs сумма трёх результатов, делим на 3
            x = xs / self.num_kernels
        # пропускаем вход через ReLU
        x = F.leaky_relu(x)
        # пропускаем вход через результирующий слой
        x = self.conv_post(x)
        # пропускаем вход через tanh
        # TODO: зачем
        x = torch.tanh(x)

        return x

    # TODO:
    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

# Дискриминатор
class DiscriminatorP(torch.nn.Module):
    # инитиализация генератора
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        # для создаваемых моделей вызывем эту строчку в конструкторе
        super(DiscriminatorP, self).__init__()
        # TODO:
        self.period = period
        # по умолчанию весы нормализуются также как и для предыдущих моделей
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # TODO: с увеличением?
        # создаеим список свёрточных слое с увеличением каналов от 1 до 1024
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

