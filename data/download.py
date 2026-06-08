# data/download.py
import torchvision
torchvision.datasets.STL10(root='./data/stl10', split='unlabeled', download=True)
torchvision.datasets.STL10(root='./data/stl10', split='train',     download=True)
torchvision.datasets.STL10(root='./data/stl10', split='test',      download=True)