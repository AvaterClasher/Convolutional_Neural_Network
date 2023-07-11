# Convolutional_Neural_Network


<div class="cell code" execution_count="36"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ZBYFRONgXujN" outputId="9cdf0161-20e9-45f6-89f8-a5b5b81b4e16">

``` python
# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Check versions
print(torch.__version__)
print(torchvision.__version__)
```

<div class="output stream stdout">

    2.0.1+cu118
    0.15.2+cu118

</div>

</div>

<div class="cell code" execution_count="37" id="bPC66_SXbsg3">

``` python
# Setup training data
from torchvision import datasets
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=torchvision.transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
```

</div>

<div class="cell code" execution_count="38"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="4W_KneC6b4Jz" outputId="31296a4b-edd8-4bbf-cc91-ae113ecd0e1f">

``` python
# See the first training example
image, label = train_data[0]
image, label
```

<div class="output execute_result" execution_count="38">

    (tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0039, 0.0000, 0.0000, 0.0510,
               0.2863, 0.0000, 0.0000, 0.0039, 0.0157, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0039, 0.0039, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0118, 0.0000, 0.1412, 0.5333,
               0.4980, 0.2431, 0.2118, 0.0000, 0.0000, 0.0000, 0.0039, 0.0118,
               0.0157, 0.0000, 0.0000, 0.0118],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0235, 0.0000, 0.4000, 0.8000,
               0.6902, 0.5255, 0.5647, 0.4824, 0.0902, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0471, 0.0392, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6078, 0.9255,
               0.8118, 0.6980, 0.4196, 0.6118, 0.6314, 0.4275, 0.2510, 0.0902,
               0.3020, 0.5098, 0.2824, 0.0588],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0039, 0.0000, 0.2706, 0.8118, 0.8745,
               0.8549, 0.8471, 0.8471, 0.6392, 0.4980, 0.4745, 0.4784, 0.5725,
               0.5529, 0.3451, 0.6745, 0.2588],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0039, 0.0039, 0.0039, 0.0000, 0.7843, 0.9098, 0.9098,
               0.9137, 0.8980, 0.8745, 0.8745, 0.8431, 0.8353, 0.6431, 0.4980,
               0.4824, 0.7686, 0.8980, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7176, 0.8824, 0.8471,
               0.8745, 0.8941, 0.9216, 0.8902, 0.8784, 0.8706, 0.8784, 0.8667,
               0.8745, 0.9608, 0.6784, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7569, 0.8941, 0.8549,
               0.8353, 0.7765, 0.7059, 0.8314, 0.8235, 0.8275, 0.8353, 0.8745,
               0.8627, 0.9529, 0.7922, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0039, 0.0118, 0.0000, 0.0471, 0.8588, 0.8627, 0.8314,
               0.8549, 0.7529, 0.6627, 0.8902, 0.8157, 0.8549, 0.8784, 0.8314,
               0.8863, 0.7725, 0.8196, 0.2039],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0235, 0.0000, 0.3882, 0.9569, 0.8706, 0.8627,
               0.8549, 0.7961, 0.7765, 0.8667, 0.8431, 0.8353, 0.8706, 0.8627,
               0.9608, 0.4667, 0.6549, 0.2196],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0157, 0.0000, 0.0000, 0.2157, 0.9255, 0.8941, 0.9020,
               0.8941, 0.9412, 0.9098, 0.8353, 0.8549, 0.8745, 0.9176, 0.8510,
               0.8510, 0.8196, 0.3608, 0.0000],
              [0.0000, 0.0000, 0.0039, 0.0157, 0.0235, 0.0275, 0.0078, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.9294, 0.8863, 0.8510, 0.8745,
               0.8706, 0.8588, 0.8706, 0.8667, 0.8471, 0.8745, 0.8980, 0.8431,
               0.8549, 1.0000, 0.3020, 0.0000],
              [0.0000, 0.0118, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.2431, 0.5686, 0.8000, 0.8941, 0.8118, 0.8353, 0.8667,
               0.8549, 0.8157, 0.8275, 0.8549, 0.8784, 0.8745, 0.8588, 0.8431,
               0.8784, 0.9569, 0.6235, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.1725, 0.3216, 0.4196,
               0.7412, 0.8941, 0.8627, 0.8706, 0.8510, 0.8863, 0.7843, 0.8039,
               0.8275, 0.9020, 0.8784, 0.9176, 0.6902, 0.7373, 0.9804, 0.9725,
               0.9137, 0.9333, 0.8431, 0.0000],
              [0.0000, 0.2235, 0.7333, 0.8157, 0.8784, 0.8667, 0.8784, 0.8157,
               0.8000, 0.8392, 0.8157, 0.8196, 0.7843, 0.6235, 0.9608, 0.7569,
               0.8078, 0.8745, 1.0000, 1.0000, 0.8667, 0.9176, 0.8667, 0.8275,
               0.8627, 0.9098, 0.9647, 0.0000],
              [0.0118, 0.7922, 0.8941, 0.8784, 0.8667, 0.8275, 0.8275, 0.8392,
               0.8039, 0.8039, 0.8039, 0.8627, 0.9412, 0.3137, 0.5882, 1.0000,
               0.8980, 0.8667, 0.7373, 0.6039, 0.7490, 0.8235, 0.8000, 0.8196,
               0.8706, 0.8941, 0.8824, 0.0000],
              [0.3843, 0.9137, 0.7765, 0.8235, 0.8706, 0.8980, 0.8980, 0.9176,
               0.9765, 0.8627, 0.7608, 0.8431, 0.8510, 0.9451, 0.2549, 0.2863,
               0.4157, 0.4588, 0.6588, 0.8588, 0.8667, 0.8431, 0.8510, 0.8745,
               0.8745, 0.8784, 0.8980, 0.1137],
              [0.2941, 0.8000, 0.8314, 0.8000, 0.7569, 0.8039, 0.8275, 0.8824,
               0.8471, 0.7255, 0.7725, 0.8078, 0.7765, 0.8353, 0.9412, 0.7647,
               0.8902, 0.9608, 0.9373, 0.8745, 0.8549, 0.8314, 0.8196, 0.8706,
               0.8627, 0.8667, 0.9020, 0.2627],
              [0.1882, 0.7961, 0.7176, 0.7608, 0.8353, 0.7725, 0.7255, 0.7451,
               0.7608, 0.7529, 0.7922, 0.8392, 0.8588, 0.8667, 0.8627, 0.9255,
               0.8824, 0.8471, 0.7804, 0.8078, 0.7294, 0.7098, 0.6941, 0.6745,
               0.7098, 0.8039, 0.8078, 0.4510],
              [0.0000, 0.4784, 0.8588, 0.7569, 0.7020, 0.6706, 0.7176, 0.7686,
               0.8000, 0.8235, 0.8353, 0.8118, 0.8275, 0.8235, 0.7843, 0.7686,
               0.7608, 0.7490, 0.7647, 0.7490, 0.7765, 0.7529, 0.6902, 0.6118,
               0.6549, 0.6941, 0.8235, 0.3608],
              [0.0000, 0.0000, 0.2902, 0.7412, 0.8314, 0.7490, 0.6863, 0.6745,
               0.6863, 0.7098, 0.7255, 0.7373, 0.7412, 0.7373, 0.7569, 0.7765,
               0.8000, 0.8196, 0.8235, 0.8235, 0.8275, 0.7373, 0.7373, 0.7608,
               0.7529, 0.8471, 0.6667, 0.0000],
              [0.0078, 0.0000, 0.0000, 0.0000, 0.2588, 0.7843, 0.8706, 0.9294,
               0.9373, 0.9490, 0.9647, 0.9529, 0.9569, 0.8667, 0.8627, 0.7569,
               0.7490, 0.7020, 0.7137, 0.7137, 0.7098, 0.6902, 0.6510, 0.6588,
               0.3882, 0.2275, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1569,
               0.2392, 0.1725, 0.2824, 0.1608, 0.1373, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000]]]),
     9)

</div>

</div>

<div class="cell code" execution_count="39"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="o0ye97vXcaAv" outputId="0c16801a-690f-4710-ffa2-c9f31ce6e372">

``` python
class_names = train_data.classes
class_names
```

<div class="output execute_result" execution_count="39">

    ['T-shirt/top',
     'Trouser',
     'Pullover',
     'Dress',
     'Coat',
     'Sandal',
     'Shirt',
     'Sneaker',
     'Bag',
     'Ankle boot']

</div>

</div>

<div class="cell code" execution_count="40"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ylPgVb7mb8iG" outputId="7e225048-c69a-4567-9cfc-6415d2292364">

``` python
class_to_idx = train_data.class_to_idx
class_to_idx
```

<div class="output execute_result" execution_count="40">

    {'T-shirt/top': 0,
     'Trouser': 1,
     'Pullover': 2,
     'Dress': 3,
     'Coat': 4,
     'Sandal': 5,
     'Shirt': 6,
     'Sneaker': 7,
     'Bag': 8,
     'Ankle boot': 9}

</div>

</div>

<div class="cell code" execution_count="41"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="V548wxSqcAsj" outputId="d116e3e8-8667-4e42-8a4f-de6b02422ab9">

``` python
train_data, test_data
```

<div class="output execute_result" execution_count="41">

    (Dataset FashionMNIST
         Number of datapoints: 60000
         Root location: data
         Split: Train
         StandardTransform
     Transform: ToTensor(),
     Dataset FashionMNIST
         Number of datapoints: 10000
         Root location: data
         Split: Test
         StandardTransform
     Transform: ToTensor())

</div>

</div>

<div class="cell code" execution_count="42"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Y1145IthcFCV" outputId="5db7b446-5fe9-4a19-d7ce-6ce980d830da">

``` python
from torch.utils.data import DataLoader

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_dataloader, test_dataloader
```

<div class="output execute_result" execution_count="42">

    (<torch.utils.data.dataloader.DataLoader at 0x7f038d895e40>,
     <torch.utils.data.dataloader.DataLoader at 0x7f038d894ca0>)

</div>

</div>

<div class="cell code" execution_count="43"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="YYyBFFX3cH0v" outputId="ef0ae9d7-b146-4278-f5ad-80a50359812f">

``` python
# Let's check out what what we've created
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")
```

<div class="output stream stdout">

    DataLoaders: (<torch.utils.data.dataloader.DataLoader object at 0x7f038d895e40>, <torch.utils.data.dataloader.DataLoader object at 0x7f038d894ca0>)
    Length of train_dataloader: 1875 batches of 32...
    Length of test_dataloader: 313 batches of 32...

</div>

</div>

<div class="cell code" execution_count="44"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="mMnwMIqTcLzi" outputId="2f34e273-7499-4cb2-eb46-749abe6de6b1">

``` python
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
```

<div class="output execute_result" execution_count="44">

    (torch.Size([32, 1, 28, 28]), torch.Size([32]))

</div>

</div>

<div class="cell code" execution_count="45"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:463}"
id="BJVBhoFOcPQ5" outputId="1ea6f79d-c7dc-4c1b-89be-2c5fa457436c">

``` python
# Show a sample
# torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
```

<div class="output stream stdout">

    Image size: torch.Size([1, 28, 28])
    Label: 1, label size: torch.Size([])

</div>

<div class="output display_data">

![](5a5385f363bd6b815284f3e3cd9c7c9d5413900f.png)

</div>

</div>

<div class="cell code" execution_count="46"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="lo2rw2QccRig" outputId="f5fd5558-e1a5-458d-c7ae-5521af6bc9f7">

``` python
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download...")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
```

<div class="output stream stdout">

    helper_functions.py already exists, skipping download...

</div>

</div>

<div class="cell code" execution_count="47" id="bmRRgrAicjhW">

``` python
from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
  """Prints difference between start and end time."""
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time
```

</div>

<div class="cell code" execution_count="48"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:36}"
id="X2YRQk0YcnoD" outputId="f63e8f2b-c8c8-4f7a-ff53-2ee5866d04d5">

``` python
# Setup device-agnostic code
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

<div class="output execute_result" execution_count="48">

``` json
{"type":"string"}
```

</div>

</div>

<div class="cell code" execution_count="49" id="JVg6XUnXcsvd">

``` python
# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
  """
  Model architecture that replicates the TinyVGG
  model from CNN explainer website.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        # Create a conv layer - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1), # values we can set ourselves in our NN's are called hyperparameters
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, # there's a trick to calculating this...
                  out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    # print(f"Output shape of conv_block_1: {x.shape}")
    x = self.conv_block_2(x)
    # print(f"Output shape of conv_block_2: {x.shape}")
    x = self.classifier(x)
    # print(f"Output shape of classifier: {x.shape}")
    return x
```

</div>

<div class="cell code" execution_count="50" id="6a5r5YZFczDT">

``` python
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
```

</div>

<div class="cell code" execution_count="51"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="f3yFz8esc1aM" outputId="07c40aed-5e04-44d9-fb7f-6070694a91b7">

``` python
rand_image_tensor = torch.randn(size=(1, 28, 28))
rand_image_tensor.shape
```

<div class="output execute_result" execution_count="51">

    torch.Size([1, 28, 28])

</div>

</div>

<div class="cell code" execution_count="52"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="HqwKO04Lc6S-" outputId="c030abf0-2d59-47de-d81d-5bcba7337623">

``` python
# Pass image through model
model_2(rand_image_tensor.unsqueeze(0).to(device))
```

<div class="output execute_result" execution_count="52">

    tensor([[ 0.0366, -0.0940,  0.0686, -0.0485,  0.0068,  0.0290,  0.0132,  0.0084,
             -0.0030, -0.0185]], device='cuda:0', grad_fn=<AddmmBackward0>)

</div>

</div>

<div class="cell code" execution_count="53"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:447}"
id="TcE81cxsc8fT" outputId="04980cbc-0906-406e-ce8a-a22d716900d3">

``` python
plt.imshow(image.squeeze(), cmap="gray")
```

<div class="output execute_result" execution_count="53">

    <matplotlib.image.AxesImage at 0x7f038cd06470>

</div>

<div class="output display_data">

![](e0915c94fc2d8214703f17b35e3f46a17c5b094b.png)

</div>

</div>

<div class="cell code" execution_count="54"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="llZrkhUgc-Tz" outputId="3baae1dd-eb6c-4f37-b786-7d6d4816b2df">

``` python
model_2.state_dict()
```

</div>

<div class="cell code" execution_count="55"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="h0k0eGwJdC8S" outputId="127b38ca-eba1-46ff-8d44-84efe1f876f6">

``` python
torch.manual_seed(42)

# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
print(f"Test image:\n {test_image}")
```

<div class="output stream stdout">

    Image batch shape: torch.Size([32, 3, 64, 64])
    Single image shape: torch.Size([3, 64, 64])
    Test image:
     tensor([[[ 1.9269,  1.4873,  0.9007,  ...,  1.8446, -1.1845,  1.3835],
             [ 1.4451,  0.8564,  2.2181,  ...,  0.3399,  0.7200,  0.4114],
             [ 1.9312,  1.0119, -1.4364,  ..., -0.5558,  0.7043,  0.7099],
             ...,
             [-0.5610, -0.4830,  0.4770,  ..., -0.2713, -0.9537, -0.6737],
             [ 0.3076, -0.1277,  0.0366,  ..., -2.0060,  0.2824, -0.8111],
             [-1.5486,  0.0485, -0.7712,  ..., -0.1403,  0.9416, -0.0118]],

            [[-0.5197,  1.8524,  1.8365,  ...,  0.8935, -1.5114, -0.8515],
             [ 2.0818,  1.0677, -1.4277,  ...,  1.6612, -2.6223, -0.4319],
             [-0.1010, -0.4388, -1.9775,  ...,  0.2106,  0.2536, -0.7318],
             ...,
             [ 0.2779,  0.7342, -0.3736,  ..., -0.4601,  0.1815,  0.1850],
             [ 0.7205, -0.2833,  0.0937,  ..., -0.1002, -2.3609,  2.2465],
             [-1.3242, -0.1973,  0.2920,  ...,  0.5409,  0.6940,  1.8563]],

            [[-0.7978,  1.0261,  1.1465,  ...,  1.2134,  0.9354, -0.0780],
             [-1.4647, -1.9571,  0.1017,  ..., -1.9986, -0.7409,  0.7011],
             [-1.3938,  0.8466, -1.7191,  ..., -1.1867,  0.1320,  0.3407],
             ...,
             [ 0.8206, -0.3745,  1.2499,  ..., -0.0676,  0.0385,  0.6335],
             [-0.5589, -0.3393,  0.2347,  ...,  2.1181,  2.4569,  1.3083],
             [-0.4092,  1.5199,  0.2401,  ..., -0.2558,  0.7870,  0.9924]]])

</div>

</div>

<div class="cell code" execution_count="56"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="FRvG4KYddHoY" outputId="60806c03-fc9f-427b-e328-dcea9fe36ff6">

``` python
test_image.shape
```

<div class="output execute_result" execution_count="56">

    torch.Size([3, 64, 64])

</div>

</div>

<div class="cell code" execution_count="57"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="t6YKaSCudKPB" outputId="b8cd7e19-402f-48ce-f8ee-835965ae8eec">

``` python
torch.manual_seed(42)
# Create a sinlge conv2d layer
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=(3, 3),
                       stride=1,
                       padding=0)

# Pass the data through the convolutional layer
conv_output = conv_layer(test_image.unsqueeze(0))
conv_output.shape
```

<div class="output execute_result" execution_count="57">

    torch.Size([1, 10, 62, 62])

</div>

</div>

<div class="cell code" execution_count="58"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="h76bK-70dM-b" outputId="2ba3cb7c-c5b5-4afc-8265-1251399e82d9">

``` python
test_image.unsqueeze(0).shape
```

<div class="output execute_result" execution_count="58">

    torch.Size([1, 3, 64, 64])

</div>

</div>

<div class="cell code" execution_count="59"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="09inJRtKdO6_" outputId="b684767e-94b2-4d17-e5e7-46dcd60d0d7e">

``` python
test_image.shape
```

<div class="output execute_result" execution_count="59">

    torch.Size([3, 64, 64])

</div>

</div>

<div class="cell code" execution_count="60"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="R0AHVvp7dUOl" outputId="9461736d-5d99-4310-9949-e4bdf58eb8ef">

``` python
# Print out original image shape without unsqueezed dimension
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(0).shape}")

# Create a sample nn.MaxPool2d layer|
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")
```

<div class="output stream stdout">

    Test image original shape: torch.Size([3, 64, 64])
    Test image with unsqueezed dimension: torch.Size([1, 3, 64, 64])
    Shape after going through conv_layer(): torch.Size([1, 10, 62, 62])
    Shape after going through conv_layer() and max_pool_layer(): torch.Size([1, 10, 31, 31])

</div>

</div>

<div class="cell code" execution_count="61"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="i4OXiPDbdZHH" outputId="27945b7b-89c8-438b-c033-924ed526a010">

``` python
torch.manual_seed(42)
# Create a random tesnor with a similar number of dimensions to our images
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"\nRandom tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n {max_pool_tensor}")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")
```

<div class="output stream stdout">


    Random tensor:
    tensor([[[[0.3367, 0.1288],
              [0.2345, 0.2303]]]])
    Random tensor shape: torch.Size([1, 1, 2, 2])

    Max pool tensor:
     tensor([[[[0.3367]]]])
    Max pool tensor shape: torch.Size([1, 1, 1, 1])

</div>

</div>

<div class="cell code" execution_count="62" id="9CMUYAlFeAou">

``` python
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
  """Performs a training with model trying to learn on data_loader."""
  train_loss, train_acc = 0, 0

  # Put model into training mode
  model.train()

  # Add a loop to loop through the training batches
  for batch, (X, y) in enumerate(data_loader):
    # Put data on target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass (outputs the raw logits from the model)
    y_pred = model(X)

    # 2. Calculate loss and accuracy (per batch)
    loss = loss_fn(y_pred, y)
    train_loss += loss # accumulate train loss
    train_acc += accuracy_fn(y_true=y,
                             y_pred=y_pred.argmax(dim=1)) # go from logits -> prediction labels

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step (update the model's parameters once *per batch*)
    optimizer.step()

  # Divide total train loss and acc by length of train dataloader
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")
```

</div>

<div class="cell code" execution_count="63" id="EhsIJgz0eC8o">

``` python
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
  """Performs a testing loop step on model going over data_loader."""
  test_loss, test_acc = 0, 0

  # Put the model in eval mode
  model.eval()

  # Turn on inference mode context manager
  with torch.inference_mode():
    for X, y in data_loader:
      # Send the data to the target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass (outputs raw logits)
      test_pred = model(X)

      # 2. Calculuate the loss/acc
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true=y,
                              y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels

    # Adjust metrics and print out
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")
```

</div>

<div class="cell code" execution_count="64" id="5M3o_ZRrdcQw">

``` python
# Setup loss function/eval metrics/optimizer
from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)
```

</div>

<div class="cell code" execution_count="92"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ixHW8f4hde7Y" outputId="31c5be96-6cea-4368-d725-01c794ac2186">

``` python
from tqdm import tqdm
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# Train and test model
epochs = 20
for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------")
  train_step(model=model_2,
             data_loader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model=model_2,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)
```

<div class="output stream stderr">

    
  0%|          | 0/20 [00:00<?, ?it/s]

</div>

<div class="output stream stdout">

    Epoch: 0
    -------
    Train loss: 0.30535 | Train acc: 88.93%

</div>

<div class="output stream stderr">

    
  5%|▌         | 1/20 [00:14<04:34, 14.47s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.32449 | Test acc: 88.08%

    Epoch: 1
    -------
    Train loss: 0.29180 | Train acc: 89.52%

</div>

<div class="output stream stderr">

    
 10%|█         | 2/20 [00:29<04:21, 14.53s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.30374 | Test acc: 89.00%

    Epoch: 2
    -------
    Train loss: 0.28143 | Train acc: 89.83%

</div>

<div class="output stream stderr">

    
 15%|█▌        | 3/20 [00:43<04:07, 14.54s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29496 | Test acc: 89.37%

    Epoch: 3
    -------
    Train loss: 0.27427 | Train acc: 90.18%

</div>

<div class="output stream stderr">

    
 20%|██        | 4/20 [00:58<03:55, 14.73s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.31116 | Test acc: 88.35%

    Epoch: 4
    -------
    Train loss: 0.26692 | Train acc: 90.26%

</div>

<div class="output stream stderr">

    
 25%|██▌       | 5/20 [01:12<03:38, 14.57s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29428 | Test acc: 89.41%

    Epoch: 5
    -------
    Train loss: 0.26032 | Train acc: 90.45%

</div>

<div class="output stream stderr">

    
 30%|███       | 6/20 [01:27<03:24, 14.63s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29493 | Test acc: 89.53%

    Epoch: 6
    -------
    Train loss: 0.25590 | Train acc: 90.70%

</div>

<div class="output stream stderr">

    
 35%|███▌      | 7/20 [01:41<03:08, 14.52s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29387 | Test acc: 89.45%

    Epoch: 7
    -------
    Train loss: 0.25286 | Train acc: 90.84%

</div>

<div class="output stream stderr">

    
 40%|████      | 8/20 [01:56<02:54, 14.51s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29094 | Test acc: 89.71%

    Epoch: 8
    -------
    Train loss: 0.24863 | Train acc: 90.84%

</div>

<div class="output stream stderr">

    
 45%|████▌     | 9/20 [02:10<02:39, 14.51s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29857 | Test acc: 89.65%

    Epoch: 9
    -------
    Train loss: 0.24446 | Train acc: 91.13%

</div>

<div class="output stream stderr">

    
 50%|█████     | 10/20 [02:25<02:26, 14.64s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.30217 | Test acc: 88.87%

    Epoch: 10
    -------
    Train loss: 0.24097 | Train acc: 91.28%

</div>

<div class="output stream stderr">

    
 55%|█████▌    | 11/20 [02:40<02:11, 14.66s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.28234 | Test acc: 90.02%

    Epoch: 11
    -------
    Train loss: 0.23862 | Train acc: 91.31%

</div>

<div class="output stream stderr">

    
 60%|██████    | 12/20 [02:55<01:58, 14.77s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.31057 | Test acc: 88.74%

    Epoch: 12
    -------
    Train loss: 0.23597 | Train acc: 91.34%

</div>

<div class="output stream stderr">

    
 65%|██████▌   | 13/20 [03:09<01:41, 14.54s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.28283 | Test acc: 89.62%

    Epoch: 13
    -------
    Train loss: 0.23392 | Train acc: 91.48%

</div>

<div class="output stream stderr">

    
 70%|███████   | 14/20 [03:23<01:26, 14.43s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29292 | Test acc: 89.72%

    Epoch: 14
    -------
    Train loss: 0.23363 | Train acc: 91.42%

</div>

<div class="output stream stderr">

    
 75%|███████▌  | 15/20 [03:37<01:11, 14.34s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29396 | Test acc: 89.25%

    Epoch: 15
    -------
    Train loss: 0.23077 | Train acc: 91.61%

</div>

<div class="output stream stderr">

    
 80%|████████  | 16/20 [03:51<00:56, 14.24s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.28371 | Test acc: 90.27%

    Epoch: 16
    -------
    Train loss: 0.22683 | Train acc: 91.62%

</div>

<div class="output stream stderr">

    
 85%|████████▌ | 17/20 [04:06<00:42, 14.27s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29502 | Test acc: 89.73%

    Epoch: 17
    -------
    Train loss: 0.22656 | Train acc: 91.77%

</div>

<div class="output stream stderr">

    
 90%|█████████ | 18/20 [04:21<00:28, 14.42s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.30488 | Test acc: 89.27%

    Epoch: 18
    -------
    Train loss: 0.22276 | Train acc: 91.88%

</div>

<div class="output stream stderr">

    
 95%|█████████▌| 19/20 [04:34<00:14, 14.28s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.28252 | Test acc: 89.87%

    Epoch: 19
    -------
    Train loss: 0.22362 | Train acc: 91.92%

</div>

<div class="output stream stderr">

    100%|██████████| 20/20 [04:49<00:00, 14.46s/it]

</div>

<div class="output stream stdout">

    Test loss: 0.29692 | Test acc: 89.42%

    Train time on cuda: 289.180 seconds

</div>

<div class="output stream stderr">

</div>

</div>

<div class="cell code" execution_count="93" id="QH2Mx2VyfOMg">

``` python
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
  """Returns a dictionary containing the results of model predicting on data_loader."""
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      # Make our data device agnostic
      X, y = X.to(device), y.to(device)
      # Make predictions
      y_pred = model(X)

      # Accumulate the loss and acc values per batch
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y,
                         y_pred=y_pred.argmax(dim=1))

    # Scale loss and acc to find the average loss/acc per batch
    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__, # only works when model was created with a class
          "model_loss": loss.item(),
          "model_acc": acc}
```

</div>

<div class="cell code" execution_count="94"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="oEsv0QD_dhhg" outputId="84402885-6554-4aae-a4b7-7b9ec2c831a2">

``` python
# Get model_2 results
model_2_results = eval_model(
     model=model_2,
     data_loader=test_dataloader,
     loss_fn=loss_fn,
     accuracy_fn=accuracy_fn,
     device=device
)

model_2_results
```

<div class="output stream stderr">

    100%|██████████| 313/313 [00:01<00:00, 200.50it/s]

</div>

<div class="output execute_result" execution_count="94">

    {'model_name': 'FashionMNISTModelV2',
     'model_loss': 0.2969161868095398,
     'model_acc': 89.41693290734824}

</div>

</div>

<div class="cell code" execution_count="95" id="uKC6BHBGeL2H">

``` python
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
  pred_probs = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample in data:
      # Prepare the sample (add a batch dimension and pass to target device)
      sample = torch.unsqueeze(sample, dim=0).to(device)

      # Forward pass (model outputs raw logits)
      pred_logit = model(sample)

      # Get prediction probability (logit -> prediction probability)
      pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

      # Get pred_prob off the GPU for further calculations
      pred_probs.append(pred_prob.cpu())

  # Stack the pred_probs to turn list into a tensor
  return torch.stack(pred_probs)
```

</div>

<div class="cell code" execution_count="96"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="c2saRohHeQ76" outputId="1545b691-c4af-4b0d-b412-66ab3bf43002">

``` python
import random
# random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)

# View the first sample shape
test_samples[0].shape
```

<div class="output execute_result" execution_count="96">

    torch.Size([1, 28, 28])

</div>

</div>

<div class="cell code" execution_count="97"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:469}"
id="yzkh6esUeWQ6" outputId="b6b1591b-0baa-4c15-b825-a39a9db17c06">

``` python
plt.imshow(test_samples[0].squeeze(), cmap="gray")
plt.title(class_names[test_labels[0]])
```

<div class="output execute_result" execution_count="97">

    Text(0.5, 1.0, 'Pullover')

</div>

<div class="output display_data">

![](2e41c446d125efdb456a0bf8e93e561d8e117bca.png)

</div>

</div>

<div class="cell code" execution_count="98"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="YlXx56SHeXts" outputId="fbb9be16-7f16-4fd6-868d-9bfc6bdf0f45">

``` python
# Make predictions
pred_probs = make_predictions(model=model_2,
                              data=test_samples)

# View first two prediction probabilities
pred_probs[:2]
```

<div class="output execute_result" execution_count="98">

    tensor([[6.1369e-04, 4.2197e-01, 4.2012e-03, 4.9927e-01, 7.2295e-02, 1.4852e-06,
             6.6723e-04, 2.4384e-05, 9.6374e-04, 1.5782e-07],
            [6.5935e-10, 2.9363e-11, 3.6637e-11, 1.7109e-10, 9.3879e-11, 1.5957e-04,
             3.0603e-09, 1.7467e-03, 1.0286e-06, 9.9809e-01]])

</div>

</div>

<div class="cell code" execution_count="99"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="dGY_Z3mreZaW" outputId="7c231ba6-b372-46b3-9316-381b69190e0a">

``` python
# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
pred_classes
```

<div class="output execute_result" execution_count="99">

    tensor([3, 9, 3, 7, 3, 7, 3, 3, 2])

</div>

</div>

<div class="cell code" execution_count="100"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="J00tiPlPeajP" outputId="7acc46db-cc3c-4ed6-ee6c-0870105a0b60">

``` python
test_labels
```

<div class="output execute_result" execution_count="100">

    [2, 9, 3, 7, 2, 7, 3, 3, 2]

</div>

</div>

<div class="cell code" execution_count="101"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:749}"
id="Qn7T8jsRecK6" outputId="b6fd4ae4-b88f-444a-8792-50c6112eafe2">

``` python
# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction (in text form, e.g "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form)
  truth_label = class_names[test_labels[i]]

  # Create a title for the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"

  # Check for equality between pred and truth and change color of title text
  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c="g") # green text if prediction same as truth
  else:
    plt.title(title_text, fontsize=10, c="r")

  plt.axis(False);
```

<div class="output display_data">

![](27dc182b1c5f5be1ea08ed27c17695a1a63f249f.png)

</div>

</div>

<div class="cell code" execution_count="102"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:66,&quot;referenced_widgets&quot;:[&quot;30f5b7af6f0547b585c801f75bccf8cc&quot;,&quot;47946bf52f9b40edb74a2d09181f1b12&quot;,&quot;cd554a1954354799b219de4d366d3de6&quot;,&quot;20ae7147545e40e2a311f6c293defb63&quot;,&quot;c7e94bccf0d8452fa1c14f129cb9c3a9&quot;,&quot;1299d8f6b82e47f99242f17e3b261829&quot;,&quot;2bec1138ecb3431ca07c4c081be75ab5&quot;,&quot;f4bf5893a1af4bf3b43ede401c785bed&quot;,&quot;898067b86d844958be9de66450d5a785&quot;,&quot;44041800c42f490db9c0bb43c35f0e72&quot;,&quot;282f60834ab8478e84dc3642d9401ddb&quot;]}"
id="KCqwfwUNefa1" outputId="3a6a8366-df78-46c3-ab3f-a02af5a054e7">

``` python
# Import tqdm.auto
from tqdm.auto import tqdm


# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions..."):
    # Send the data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_2(X)
    # Turn predictions from logits -> prediction probabilities -> prediction labels
    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
    # Put prediction on CPU for evaluation
    y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
# print(y_preds)
y_pred_tensor = torch.cat(y_preds)
y_pred_tensor
```

<div class="output display_data">

``` json
{"model_id":"30f5b7af6f0547b585c801f75bccf8cc","version_major":2,"version_minor":0}
```

</div>

<div class="output execute_result" execution_count="102">

    tensor([9, 2, 1,  ..., 8, 1, 0])

</div>

</div>

<div class="cell code" execution_count="103"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="W1EOTqzbehM4" outputId="849be0b5-6382-42d4-c037-e0e915a4bf2b">

``` python
len(y_pred_tensor)
```

<div class="output execute_result" execution_count="103">

    10000

</div>

</div>

<div class="cell code" execution_count="104"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="OBQU0PyRehrz" outputId="0b7465f8-bf8e-4264-93da-e71c4fc9b305">

``` python
# See if required packages are installed and if not, install them...
try:
  import torchmetrics, mlxtend
  print(f"mlxtend version: {mlxtend.__version__}")
  assert int(mlxtend.__version__.split(".")[1] >= 19, "mlxtend version should be 0.19.0 or higher")
except:
  !pip install torchmetrics -U mlxtend
  import torchmetrics, mlxtend
  print(f"mlxtend version: {mlxtend.__version__}")
```

<div class="output stream stdout">

    mlxtend version: 0.22.0
    Requirement already satisfied: torchmetrics in /usr/local/lib/python3.10/dist-packages (1.0.0)
    Requirement already satisfied: mlxtend in /usr/local/lib/python3.10/dist-packages (0.22.0)
    Requirement already satisfied: numpy>1.20.0 in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (1.22.4)
    Requirement already satisfied: torch>=1.8.1 in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (2.0.1+cu118)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (23.1)
    Requirement already satisfied: lightning-utilities>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (0.9.0)
    Requirement already satisfied: scipy>=1.2.1 in /usr/local/lib/python3.10/dist-packages (from mlxtend) (1.10.1)
    Requirement already satisfied: pandas>=0.24.2 in /usr/local/lib/python3.10/dist-packages (from mlxtend) (1.5.3)
    Requirement already satisfied: scikit-learn>=1.0.2 in /usr/local/lib/python3.10/dist-packages (from mlxtend) (1.2.2)
    Requirement already satisfied: matplotlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from mlxtend) (3.7.1)
    Requirement already satisfied: joblib>=0.13.2 in /usr/local/lib/python3.10/dist-packages (from mlxtend) (1.2.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from mlxtend) (67.7.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from lightning-utilities>=0.7.0->torchmetrics) (4.6.3)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (4.40.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (1.4.4)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (8.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (3.1.0)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->mlxtend) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.24.2->mlxtend) (2022.7.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=1.0.2->mlxtend) (3.1.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.1->torchmetrics) (3.12.2)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.1->torchmetrics) (1.11.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.1->torchmetrics) (3.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.1->torchmetrics) (3.1.2)
    Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.1->torchmetrics) (2.0.0)
    Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.8.1->torchmetrics) (3.25.2)
    Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.8.1->torchmetrics) (16.0.6)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.0.0->mlxtend) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.1->torchmetrics) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.8.1->torchmetrics) (1.3.0)
    mlxtend version: 0.22.0

</div>

</div>

<div class="cell code" execution_count="105"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="91thQHnzelCa" outputId="7b4d9f2b-eb05-4352-edfc-1ab6916a8751">

``` python
import mlxtend
print(mlxtend.__version__)
```

<div class="output stream stdout">

    0.22.0

</div>

</div>

<div class="cell code" execution_count="106"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="LMuJh3mqemcS" outputId="ce04f327-93d9-45b6-bff9-15b732c3f3fe">

``` python
class_names
```

<div class="output execute_result" execution_count="106">

    ['T-shirt/top',
     'Trouser',
     'Pullover',
     'Dress',
     'Coat',
     'Sandal',
     'Shirt',
     'Sneaker',
     'Bag',
     'Ankle boot']

</div>

</div>

<div class="cell code" execution_count="107"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="nbEtI8y5enHg" outputId="29bb235a-300c-42bc-bafa-e8c59c1da874">

``` python
y_pred_tensor[:10]
```

<div class="output execute_result" execution_count="107">

    tensor([9, 2, 1, 1, 6, 1, 2, 6, 5, 7])

</div>

</div>

<div class="cell code" execution_count="108"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="XXxC2rmeeolW" outputId="549a63cc-33b2-4e79-95e7-a87027218386">

``` python
test_data.targets
```

<div class="output execute_result" execution_count="108">

    tensor([9, 2, 1,  ..., 8, 1, 5])

</div>

</div>

<div class="cell code" execution_count="109"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:667}"
id="br5crtYqeqCI" outputId="81fb293f-579a-47bc-b037-53eb12e45ba9">

``` python
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);
```

<div class="output display_data">

![](8ac3425c168afcb0fbc30e3dc0b791a4f9dbf0ae.png)

</div>

</div>

<div class="cell code" execution_count="110"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="6ww6aanJer5E" outputId="26fd81cc-be7a-43e9-dbac-5a6de4ac494c">

``` python
from pathlib import Path

# Create model dictory path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# Create model save
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)
```

<div class="output stream stdout">

    Saving model to: models/03_pytorch_computer_vision_model_2.pth

</div>

</div>

<div class="cell code" execution_count="111" id="5dr5Hk7te2Ki">

``` python
image_shape = [1, 28, 28]
```

</div>

<div class="cell code" execution_count="112"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="vfBiFYRye3nt" outputId="c283f2e6-14ec-4bef-d967-4a36b26dd925">

``` python
# Create a new instance
torch.manual_seed(42)

loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                                     hidden_units=10,
                                     output_shape=len(class_names))

# Load in the save state_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send the model to the target device
loaded_model_2.to(device)
```

<div class="output execute_result" execution_count="112">

    FashionMNISTModelV2(
      (conv_block_1): Sequential(
        (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv_block_2): Sequential(
        (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=490, out_features=10, bias=True)
      )
    )

</div>

</div>

<div class="cell code" execution_count="113"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:101,&quot;referenced_widgets&quot;:[&quot;ba436e8d468741b4bc84b4268eab2a5e&quot;,&quot;9117a08e2bd74cd4ba031bcd7e3ea285&quot;,&quot;98bb1bbe459c4b7a8493f206e5f81c0d&quot;,&quot;5e8abc6893f6457697d74abbc5a93b99&quot;,&quot;36e70438d1f74720b4159dcb6e2de9e3&quot;,&quot;ae487ead2831479cbcbdbcef6a21c357&quot;,&quot;f3b079c6091f400a8ff6fbccece3c63b&quot;,&quot;9f6064953cf5421dbb55a63dda10fc02&quot;,&quot;75340e52f0914fa48e4f819bcad589ec&quot;,&quot;43034829643b4e6d92d5ae6534e08ced&quot;,&quot;0ca21a72546d44988f8cb98a21a4fb50&quot;]}"
id="usTa2V89e5PC" outputId="6da1b11c-8318-4e0f-c751-0eb5cb74f947">

``` python
# Evaluate loaded model
torch.manual_seed(42)

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)

loaded_model_2_results
```

<div class="output display_data">

``` json
{"model_id":"ba436e8d468741b4bc84b4268eab2a5e","version_major":2,"version_minor":0}
```

</div>

<div class="output execute_result" execution_count="113">

    {'model_name': 'FashionMNISTModelV2',
     'model_loss': 0.2969161868095398,
     'model_acc': 89.41693290734824}

</div>

</div>

<div class="cell code" execution_count="114"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="PCAOcPQPe6T0" outputId="c8a76e03-0a4c-4be0-f237-a663f883438a">

``` python
model_2_results
```

<div class="output execute_result" execution_count="114">

    {'model_name': 'FashionMNISTModelV2',
     'model_loss': 0.2969161868095398,
     'model_acc': 89.41693290734824}

</div>

</div>

<div class="cell code" execution_count="115"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="zXY2uNu2e7ee" outputId="88bd7b02-c756-49e3-8c95-010b41c5f3bb">

``` python
# Check if model results are close to each other
torch.isclose(torch.tensor(model_2_results["model_loss"]),
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-02)
```

<div class="output execute_result" execution_count="115">

    tensor(True)

</div>

</div>
