import torch
import torch.nn as nn
import torch.nn.functional as F

# ================ Transpose Layer ================ #
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class ResNet3LN(nn.Module):
	def __init__(self, in_channels=1, mid_channels=64, kernel_sizes=[7, 5, 3], nb_class=2):
		super().__init__()

		
		self.input_args = {
			'in_channels': in_channels,
			'nb_class': nb_class
		}

		self.layers = nn.Sequential(*[
			ResNetBlock(in_channels=in_channels, out_channels=mid_channels, kernel_sizes=kernel_sizes),
			ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2, kernel_sizes=kernel_sizes),
			ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernel_sizes=kernel_sizes),

		])
		self.linear = nn.Linear(mid_channels * 2, nb_class)

	def forward(self, x):
		x = self.layers(x)
		return self.linear(x.mean(dim=-1))


class ResNetBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_sizes):
		super().__init__()

		channels = [in_channels, out_channels, out_channels, out_channels]

		self.layers = nn.Sequential(*[
			ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
					  kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
		])

		self.match_channels = False
		if in_channels != out_channels:
			self.match_channels = True
			self.residual = nn.Sequential(*[
				nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
				Transpose(1,2), 
				nn.LayerNorm(out_channels, eps=1e-05),
				Transpose(1,2),
			])

	def forward(self, x):
		if self.match_channels:
			return self.layers(x) + self.residual(x)
		return self.layers(x)
    

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, padding_mode='replicate'),
			nn.ReLU(),
			Transpose(1,2), 
			nn.LayerNorm(out_channels, eps=1e-05),
			Transpose(1,2),
		)

	def forward(self, x):
		return self.layers(x)