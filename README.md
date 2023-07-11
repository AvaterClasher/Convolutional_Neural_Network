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

<div class="output execute_result" execution_count="54">

    OrderedDict([('conv_block_1.0.weight',
                  tensor([[[[ 0.2548,  0.2767, -0.0781],
                            [ 0.3062, -0.0730,  0.0673],
                            [-0.1623,  0.1958,  0.2938]]],
                  
                  
                          [[[-0.2445,  0.2897,  0.0624],
                            [ 0.2463,  0.0451,  0.1607],
                            [-0.0471,  0.2570,  0.0493]]],
                  
                  
                          [[[-0.1556,  0.0850, -0.1536],
                            [-0.0391, -0.1354,  0.2211],
                            [-0.2631, -0.1537, -0.0941]]],
                  
                  
                          [[[-0.2004,  0.0315, -0.3292],
                            [ 0.3010, -0.2832,  0.2573],
                            [ 0.0555, -0.1082,  0.2060]]],
                  
                  
                          [[[ 0.0520,  0.2693,  0.0364],
                            [-0.1051,  0.0896, -0.0904],
                            [ 0.1403,  0.2976,  0.1927]]],
                  
                  
                          [[[-0.1457,  0.1924,  0.0596],
                            [ 0.1693, -0.2032, -0.3300],
                            [-0.1288, -0.2557,  0.2735]]],
                  
                  
                          [[[ 0.0960,  0.1381,  0.1054],
                            [-0.0058,  0.2609, -0.2368],
                            [ 0.0210, -0.2275,  0.1028]]],
                  
                  
                          [[[-0.1148,  0.1021, -0.0694],
                            [ 0.2765, -0.1976, -0.1988],
                            [-0.1988,  0.2998,  0.1111]]],
                  
                  
                          [[[ 0.3208, -0.2751, -0.3306],
                            [-0.2608, -0.2242,  0.1350],
                            [ 0.1194,  0.2770, -0.1721]]],
                  
                  
                          [[[-0.2272,  0.1769, -0.1347],
                            [ 0.2023, -0.0791,  0.1907],
                            [-0.2590, -0.1682,  0.1016]]]], device='cuda:0')),
                 ('conv_block_1.0.bias',
                  tensor([ 0.0705, -0.0850,  0.1987,  0.2266, -0.2417, -0.1780,  0.3052, -0.1125,
                          -0.1182, -0.3225], device='cuda:0')),
                 ('conv_block_1.2.weight',
                  tensor([[[[-0.0604,  0.0263, -0.0139],
                            [-0.0765,  0.0025, -0.0720],
                            [-0.0894, -0.0580, -0.0923]],
                  
                           [[-0.0671,  0.1054,  0.0199],
                            [ 0.0325, -0.0983, -0.0692],
                            [-0.0351,  0.0165, -0.0928]],
                  
                           [[-0.0454, -0.0631,  0.0003],
                            [-0.0392, -0.0073, -0.0714],
                            [-0.0724, -0.0615, -0.0361]],
                  
                           [[-0.0832,  0.0884, -0.0209],
                            [ 0.0907,  0.0328, -0.0893],
                            [ 0.0729, -0.0290, -0.0404]],
                  
                           [[-0.0875, -0.1048,  0.0302],
                            [-0.0230,  0.0410, -0.0865],
                            [ 0.0783, -0.0774, -0.0182]],
                  
                           [[ 0.0220,  0.0544,  0.0851],
                            [ 0.0960, -0.0836,  0.0265],
                            [-0.0453, -0.0116, -0.0789]],
                  
                           [[ 0.0960, -0.0774,  0.0563],
                            [ 0.0370,  0.0343, -0.0570],
                            [ 0.0958,  0.0232,  0.0136]],
                  
                           [[-0.0929,  0.0442, -0.0158],
                            [-0.0483,  0.0905,  0.0235],
                            [-0.0583, -0.0534, -0.0050]],
                  
                           [[ 0.0589, -0.0269, -0.0601],
                            [-0.0361, -0.0787,  0.0376],
                            [ 0.0816, -0.0992,  0.0245]],
                  
                           [[ 0.0545,  0.0191, -0.0375],
                            [ 0.0550,  0.0554,  0.0394],
                            [-0.0185, -0.0279,  0.0113]]],
                  
                  
                          [[[-0.0186, -0.0314,  0.0674],
                            [ 0.0906, -0.0104, -0.0236],
                            [ 0.0015, -0.0063,  0.0253]],
                  
                           [[ 0.0295, -0.0957, -0.0389],
                            [ 0.0888,  0.0411, -0.0052],
                            [-0.0636, -0.0645, -0.0944]],
                  
                           [[-0.0344,  0.0356,  0.0672],
                            [ 0.0487, -0.0932, -0.0634],
                            [-0.0166,  0.1020,  0.0152]],
                  
                           [[-0.0273,  0.0436, -0.0401],
                            [-0.0682,  0.0769, -0.0479],
                            [-0.0211, -0.1049,  0.0705]],
                  
                           [[ 0.0799,  0.0384, -0.0735],
                            [-0.1040, -0.0856,  0.0786],
                            [ 0.0506,  0.0887,  0.0552]],
                  
                           [[ 0.0267, -0.0010, -0.0802],
                            [-0.0903, -0.0986,  0.0432],
                            [-0.0518, -0.0212, -0.0607]],
                  
                           [[-0.0192, -0.0742, -0.0689],
                            [ 0.0350, -0.0313,  0.0651],
                            [-0.0338, -0.0773, -0.0186]],
                  
                           [[-0.0511, -0.0322, -0.1003],
                            [ 0.0590, -0.0734,  0.0530],
                            [ 0.0478,  0.0753, -0.0809]],
                  
                           [[ 0.0758, -0.0498,  0.0391],
                            [ 0.0990, -0.0149, -0.0008],
                            [-0.0243, -0.0880,  0.0506]],
                  
                           [[-0.1046,  0.0654,  0.0789],
                            [ 0.0997, -0.0249, -0.0866],
                            [ 0.0237,  0.0582, -0.1049]]],
                  
                  
                          [[[-0.0239, -0.0632, -0.0092],
                            [-0.0519, -0.0431, -0.0335],
                            [-0.1002,  0.0865,  0.0884]],
                  
                           [[-0.0165, -0.0120, -0.0430],
                            [-0.0952, -0.1026,  0.0392],
                            [-0.0579, -0.0678, -0.0082]],
                  
                           [[-0.0351, -0.0341,  0.0034],
                            [-0.0224, -0.0363, -0.0505],
                            [-0.0858,  0.0884, -0.0422]],
                  
                           [[ 0.0279, -0.0366,  0.0086],
                            [ 0.0983,  0.0486, -0.0913],
                            [ 0.0418,  0.1001,  0.0277]],
                  
                           [[ 0.0707,  0.1039, -0.0162],
                            [ 0.0219, -0.0733, -0.0217],
                            [ 0.0781,  0.0540, -0.0667]],
                  
                           [[-0.0845, -0.0720, -0.1040],
                            [-0.0813, -0.0261,  0.0711],
                            [ 0.0176, -0.0802, -0.0846]],
                  
                           [[ 0.0524, -0.0784, -0.0130],
                            [ 0.0506, -0.0488, -0.0115],
                            [-0.0092, -0.0249, -0.0534]],
                  
                           [[-0.0940, -0.0852, -0.0564],
                            [ 0.1018, -0.0509, -0.0708],
                            [ 0.0256,  0.0291,  0.0578]],
                  
                           [[ 0.0801,  0.0587, -0.1045],
                            [ 0.0093,  0.0639, -0.0097],
                            [-0.0621,  0.1005, -0.0394]],
                  
                           [[-0.0600, -0.0950,  0.0047],
                            [ 0.0467,  0.0233,  0.0208],
                            [-0.0799, -0.0984,  0.0019]]],
                  
                  
                          [[[ 0.0961,  0.0608, -0.0614],
                            [-0.0137, -0.0777, -0.0509],
                            [ 0.0191,  0.0574,  0.0873]],
                  
                           [[-0.0968,  0.0705, -0.0743],
                            [ 0.0395,  0.0892,  0.0015],
                            [ 0.0959, -0.0898, -0.0403]],
                  
                           [[ 0.0615, -0.0230, -0.0216],
                            [-0.0439,  0.0727,  0.0517],
                            [ 0.0338, -0.0592, -0.0856]],
                  
                           [[ 0.0114,  0.0312, -0.0487],
                            [-0.0295,  0.0712,  0.0084],
                            [ 0.0048, -0.0259, -0.0955]],
                  
                           [[-0.0991, -0.0504, -0.0536],
                            [ 0.0328, -0.0307, -0.0412],
                            [ 0.1005,  0.0367,  0.0751]],
                  
                           [[-0.0510, -0.0431,  0.0387],
                            [-0.0702, -0.0689, -0.0051],
                            [-0.0386, -0.0790,  0.0625]],
                  
                           [[ 0.0848,  0.0171, -0.0184],
                            [-0.0976, -0.0384,  0.0268],
                            [ 0.0497, -0.0133, -0.0417]],
                  
                           [[ 0.0587, -0.0839,  0.0666],
                            [-0.0409,  0.0016, -0.0208],
                            [ 0.0128, -0.0319,  0.0766]],
                  
                           [[-0.0027,  0.0823,  0.1013],
                            [-0.0514, -0.0769,  0.0846],
                            [ 0.0826, -0.0805, -0.0081]],
                  
                           [[-0.1039, -0.0863,  0.0204],
                            [ 0.0280,  0.0223, -0.0287],
                            [ 0.0972,  0.0151, -0.0622]]],
                  
                  
                          [[[-0.0060,  0.0253,  0.0369],
                            [-0.0745,  0.0395, -0.0539],
                            [-0.0876, -0.0576,  0.1017]],
                  
                           [[ 0.0901,  0.0944,  0.0619],
                            [ 0.0796, -0.0141, -0.0580],
                            [ 0.0527, -0.0546, -0.0711]],
                  
                           [[-0.0337,  0.0221,  0.0543],
                            [-0.0409, -0.0620,  0.0142],
                            [-0.0621, -0.0686,  0.0549]],
                  
                           [[-0.0177,  0.0963,  0.1025],
                            [ 0.0315,  0.0363,  0.0243],
                            [ 0.0017, -0.0077,  0.0014]],
                  
                           [[ 0.0394,  0.0980, -0.0273],
                            [-0.0446, -0.0255, -0.0509],
                            [ 0.0179,  0.0787,  0.0824]],
                  
                           [[ 0.0484, -0.0776, -0.0566],
                            [-0.0232, -0.0194,  0.0087],
                            [-0.0968,  0.0328, -0.0804]],
                  
                           [[-0.0667, -0.0876,  0.0918],
                            [-0.0998,  0.0795, -0.0035],
                            [-0.0123,  0.0659, -0.0097]],
                  
                           [[ 0.0661,  0.0762, -0.0915],
                            [ 0.0406,  0.0199,  0.0227],
                            [ 0.0154,  0.0288, -0.0507]],
                  
                           [[-0.0135,  0.1002,  0.0708],
                            [-0.0040, -0.0991,  0.0046],
                            [-0.0718,  0.0857, -0.0640]],
                  
                           [[-0.0076, -0.0234,  0.0188],
                            [ 0.0992,  0.0100,  0.0610],
                            [ 0.0818,  0.0851, -0.0364]]],
                  
                  
                          [[[-0.0236,  0.0508, -0.0288],
                            [ 0.0494, -0.0230, -0.0715],
                            [ 0.0429,  0.0162,  0.0470]],
                  
                           [[ 0.1047,  0.0720,  0.0999],
                            [ 0.0056, -0.0907, -0.0739],
                            [-0.0655, -0.0929, -0.0528]],
                  
                           [[-0.0970, -0.0973, -0.0630],
                            [-0.1039, -0.0647,  0.0402],
                            [ 0.0879, -0.0314, -0.0307]],
                  
                           [[ 0.0563, -0.0520, -0.0498],
                            [ 0.0649, -0.0918,  0.0129],
                            [ 0.0931,  0.0181,  0.0287]],
                  
                           [[-0.0614, -0.0015,  0.0058],
                            [ 0.0259,  0.0410,  0.0916],
                            [-0.0805,  0.0032, -0.0527]],
                  
                           [[-0.0834, -0.0084, -0.0928],
                            [ 0.0736,  0.0122, -0.0568],
                            [ 0.0551, -0.0998, -0.0408]],
                  
                           [[-0.0205, -0.0896, -0.0670],
                            [-0.0172,  0.0800,  0.1018],
                            [ 0.0671, -0.0629, -0.0690]],
                  
                           [[ 0.0920,  0.0373,  0.0028],
                            [ 0.0143, -0.0847, -0.0352],
                            [ 0.1015, -0.0260, -0.0053]],
                  
                           [[-0.0875, -0.0590, -0.0022],
                            [-0.0655, -0.0131,  0.0429],
                            [-0.1031,  0.0313, -0.0697]],
                  
                           [[-0.0514,  0.0405,  0.0838],
                            [-0.0288, -0.0433, -0.0953],
                            [-0.0544, -0.0923, -0.0241]]],
                  
                  
                          [[[ 0.0215, -0.0988,  0.0920],
                            [ 0.0661, -0.1032, -0.0503],
                            [ 0.0344, -0.0217, -0.0115]],
                  
                           [[-0.0476,  0.0847, -0.0589],
                            [ 0.0874,  0.0068,  0.0212],
                            [ 0.0822, -0.0174, -0.0600]],
                  
                           [[-0.0170,  0.0855, -0.0782],
                            [ 0.0239, -0.1036,  0.0553],
                            [ 0.0389,  0.0045,  0.0452]],
                  
                           [[ 0.0001,  0.0583, -0.0834],
                            [-0.0155,  0.0468,  0.1050],
                            [ 0.0537, -0.0767,  0.0811]],
                  
                           [[-0.0235, -0.0225, -0.0958],
                            [-0.0166,  0.0746,  0.0147],
                            [-0.0614,  0.0324, -0.0338]],
                  
                           [[ 0.0962, -0.0915, -0.0333],
                            [-0.1018, -0.0415,  0.0332],
                            [ 0.1015,  0.0177,  0.1033]],
                  
                           [[ 0.0206,  0.0609,  0.0845],
                            [ 0.0881, -0.0590,  0.0969],
                            [ 0.0639, -0.0493, -0.0503]],
                  
                           [[-0.0884,  0.0265, -0.0854],
                            [ 0.0445,  0.0333, -0.0916],
                            [ 0.0287, -0.0086,  0.0482]],
                  
                           [[ 0.0605, -0.1048,  0.0967],
                            [ 0.0884,  0.0419, -0.0963],
                            [-0.0377, -0.0305, -0.0271]],
                  
                           [[ 0.0594,  0.0383,  0.0835],
                            [-0.0395,  0.0355,  0.0375],
                            [-0.0878, -0.1022, -0.0547]]],
                  
                  
                          [[[ 0.0722, -0.0992, -0.0918],
                            [ 0.0591,  0.0569,  0.0867],
                            [-0.0796, -0.0771,  0.0541]],
                  
                           [[ 0.0917,  0.0631,  0.0165],
                            [ 0.0347,  0.1000, -0.0680],
                            [-0.0479,  0.0737, -0.0721]],
                  
                           [[-0.0581,  0.0769,  0.0333],
                            [ 0.0341, -0.0447, -0.0015],
                            [ 0.0965, -0.0633,  0.0008]],
                  
                           [[ 0.0501, -0.0728,  0.1024],
                            [-0.0527, -0.0253, -0.0285],
                            [-0.0687, -0.1034,  0.0594]],
                  
                           [[ 0.0280, -0.0987, -0.0678],
                            [ 0.1042,  0.0403,  0.0423],
                            [-0.0631, -0.0462, -0.0159]],
                  
                           [[-0.0193, -0.0722,  0.0087],
                            [ 0.0105, -0.0133,  0.0146],
                            [-0.0418,  0.0274,  0.0398]],
                  
                           [[-0.0555, -0.1045,  0.0552],
                            [ 0.0251, -0.0536,  0.1016],
                            [-0.0477,  0.0712,  0.0535]],
                  
                           [[-0.0884,  0.0680, -0.0969],
                            [-0.0584, -0.0176, -0.0711],
                            [ 0.1030, -0.0211,  0.0419]],
                  
                           [[-0.0941,  0.0607, -0.0328],
                            [-0.0802,  0.0154,  0.0511],
                            [ 0.0912, -0.0644, -0.0519]],
                  
                           [[ 0.0203,  0.0286,  0.0405],
                            [ 0.0579, -0.0239,  0.0586],
                            [ 0.0777, -0.0275,  0.0750]]],
                  
                  
                          [[[ 0.0515,  0.0930, -0.0599],
                            [-0.0521, -0.0305,  0.0053],
                            [ 0.0633, -0.0602,  0.0528]],
                  
                           [[-0.0378,  0.0637, -0.0050],
                            [-0.0923, -0.0580, -0.0763],
                            [ 0.0523, -0.0707, -0.0088]],
                  
                           [[ 0.0227, -0.0578,  0.0304],
                            [-0.1029, -0.0754, -0.0955],
                            [-0.0319, -0.0384,  0.0151]],
                  
                           [[-0.0195,  0.0496,  0.0966],
                            [ 0.0378, -0.0415, -0.0987],
                            [ 0.0382, -0.0522,  0.0536]],
                  
                           [[ 0.0705,  0.0407,  0.0989],
                            [ 0.1001,  0.0223, -0.0768],
                            [ 0.0942, -0.0500, -0.0498]],
                  
                           [[ 0.0882,  0.0817,  0.0318],
                            [ 0.0066, -0.0887, -0.0109],
                            [ 0.1011,  0.0268,  0.0090]],
                  
                           [[-0.0219, -0.0368,  0.0628],
                            [ 0.0065,  0.0686, -0.0187],
                            [ 0.0461,  0.0435,  0.0168]],
                  
                           [[ 0.0662,  0.0661,  0.0977],
                            [ 0.0810, -0.0270, -0.0892],
                            [ 0.0193, -0.0009, -0.0275]],
                  
                           [[-0.0177,  0.0050,  0.0769],
                            [ 0.0329, -0.0374, -0.0433],
                            [-0.0261, -0.0407,  0.0948]],
                  
                           [[ 0.0558,  0.0952,  0.0003],
                            [ 0.0213,  0.0366, -0.0998],
                            [ 0.0094, -0.0071, -0.0591]]],
                  
                  
                          [[[-0.0818,  0.0933,  0.0857],
                            [ 0.0489,  0.1006, -0.0428],
                            [-0.0182,  0.0399, -0.0174]],
                  
                           [[-0.0207, -0.0871,  0.0283],
                            [-0.0637,  0.0038,  0.1028],
                            [-0.0324, -0.0332,  0.0636]],
                  
                           [[-0.0388, -0.0091,  0.0984],
                            [-0.0432, -0.0754, -0.0590],
                            [-0.0292, -0.0500, -0.0547]],
                  
                           [[ 0.0426,  0.0179, -0.0337],
                            [-0.0819, -0.0332, -0.0445],
                            [-0.0343, -0.0951,  0.0227]],
                  
                           [[-0.0774, -0.0821, -0.0861],
                            [ 0.0440, -0.0635, -0.0435],
                            [ 0.0826,  0.0560,  0.0604]],
                  
                           [[-0.1001, -0.0756, -0.0398],
                            [ 0.0871,  0.0108, -0.0788],
                            [ 0.0007, -0.0819, -0.0231]],
                  
                           [[-0.0290,  0.0912,  0.0326],
                            [-0.0184,  0.0178, -0.0304],
                            [ 0.0414,  0.0417,  0.0283]],
                  
                           [[-0.0411,  0.0899, -0.0152],
                            [-0.0410,  0.0660,  0.0859],
                            [ 0.1049,  0.0312, -0.0359]],
                  
                           [[ 0.0535,  0.0904, -0.1034],
                            [-0.0131, -0.0719,  0.0196],
                            [ 0.0436, -0.0218, -0.0088]],
                  
                           [[ 0.0474, -0.0177, -0.0885],
                            [ 0.0843, -0.0531, -0.0116],
                            [ 0.0099, -0.0063, -0.0992]]]], device='cuda:0')),
                 ('conv_block_1.2.bias',
                  tensor([ 0.0484, -0.0479, -0.0547,  0.0252, -0.0550, -0.0487, -0.0355, -0.0396,
                          -0.0440, -0.0284], device='cuda:0')),
                 ('conv_block_2.0.weight',
                  tensor([[[[ 2.7393e-02, -8.5299e-02, -6.3802e-02],
                            [ 1.5381e-03,  1.4659e-02,  5.8217e-02],
                            [-7.4044e-02,  3.3646e-02,  5.9914e-02]],
                  
                           [[ 5.8530e-02, -9.8180e-02, -4.0225e-02],
                            [-9.0606e-02, -6.6704e-02,  5.8711e-02],
                            [-1.5740e-02,  4.4769e-02, -6.1876e-02]],
                  
                           [[ 1.6018e-02, -6.3758e-02,  5.2693e-02],
                            [-4.6104e-02, -2.6432e-02, -9.1456e-02],
                            [ 3.4822e-04,  1.0008e-01,  5.1163e-02]],
                  
                           [[-5.6240e-02,  1.4176e-03, -1.1558e-02],
                            [-8.4862e-02,  8.2650e-02,  1.6993e-03],
                            [ 2.2199e-02, -4.2567e-02, -4.9323e-02]],
                  
                           [[ 1.7381e-02,  3.8971e-02,  2.3643e-02],
                            [-5.0801e-02,  1.0234e-01, -1.5517e-02],
                            [-6.4554e-02, -4.9301e-02,  1.0377e-01]],
                  
                           [[ 5.0766e-06, -1.4309e-02, -4.3867e-02],
                            [-2.7633e-02, -8.8779e-02, -8.3767e-02],
                            [ 6.1695e-02,  9.0172e-02,  1.0059e-01]],
                  
                           [[-7.6099e-02,  5.7012e-02, -6.5245e-02],
                            [ 6.2883e-02,  7.6058e-02,  8.1573e-02],
                            [ 7.5900e-02,  6.5941e-02,  2.0516e-03]],
                  
                           [[ 4.8434e-02, -3.7712e-02,  4.5899e-02],
                            [-3.3879e-02, -1.7700e-03, -9.1746e-02],
                            [-2.7562e-02, -5.5432e-02, -3.5557e-02]],
                  
                           [[-6.7313e-02, -9.4810e-02,  6.8639e-03],
                            [ 6.8408e-02,  9.6001e-02,  6.1512e-02],
                            [-5.4638e-02, -1.0425e-01,  3.9983e-02]],
                  
                           [[ 5.9062e-02, -9.0495e-02,  3.7798e-02],
                            [ 8.9121e-02,  6.3853e-03, -6.3505e-02],
                            [ 8.6423e-02,  4.5011e-02,  6.9802e-02]]],
                  
                  
                          [[[-7.1287e-02,  6.1342e-02, -7.2002e-02],
                            [ 1.0430e-01, -4.4662e-02,  6.3516e-02],
                            [ 2.1107e-02,  2.7935e-02, -1.6165e-02]],
                  
                           [[ 4.3295e-02, -4.3932e-02, -9.9357e-02],
                            [-4.0499e-02,  8.2592e-02, -2.7751e-02],
                            [ 3.3132e-02, -3.8973e-02,  7.9073e-02]],
                  
                           [[ 6.3086e-02,  3.7211e-02, -5.3881e-02],
                            [-8.6133e-02,  3.9686e-03, -6.1839e-02],
                            [ 8.6667e-02, -1.0130e-01,  4.7104e-02]],
                  
                           [[ 1.0508e-01,  5.2792e-02,  3.5942e-02],
                            [-1.0142e-01,  1.0139e-01, -1.8030e-02],
                            [-9.8495e-02,  1.0406e-01, -4.2894e-02]],
                  
                           [[-7.4575e-03,  9.6479e-02, -7.3070e-02],
                            [-7.4576e-02,  1.7141e-02, -1.4109e-02],
                            [ 2.4280e-02, -8.8407e-02,  3.1524e-03]],
                  
                           [[-4.6882e-02, -5.1820e-02, -9.6517e-02],
                            [ 5.5890e-02,  2.0306e-02, -8.9118e-02],
                            [ 8.3648e-02,  3.1794e-02,  1.9560e-02]],
                  
                           [[-6.1890e-02,  1.5896e-02,  1.0157e-01],
                            [ 7.2299e-02, -8.2100e-02,  9.6220e-02],
                            [ 8.1702e-03,  5.0698e-02,  8.1869e-02]],
                  
                           [[ 8.9862e-02, -8.2170e-02,  9.2303e-02],
                            [-7.1591e-02,  7.9021e-03, -7.3656e-02],
                            [-2.3109e-02, -4.7901e-03, -1.2611e-02]],
                  
                           [[-1.6652e-02,  8.3137e-03,  1.0398e-01],
                            [ 6.1244e-02,  5.8973e-02,  4.2190e-02],
                            [ 8.1606e-02, -4.8645e-03,  8.3813e-03]],
                  
                           [[ 2.1693e-02, -9.1931e-02, -8.4913e-02],
                            [ 1.2923e-02, -4.1241e-02, -1.9342e-03],
                            [-2.4187e-02,  1.6408e-02,  6.8581e-02]]],
                  
                  
                          [[[-3.4958e-02,  8.4418e-02,  8.3227e-02],
                            [-8.0901e-02, -8.1400e-02, -8.5284e-02],
                            [-5.7766e-02, -4.1033e-02, -7.9341e-03]],
                  
                           [[-2.5635e-02, -5.3258e-02, -3.3488e-02],
                            [-3.8131e-02,  1.0341e-01, -3.9068e-02],
                            [-7.5473e-02,  4.3818e-02, -6.0886e-03]],
                  
                           [[ 8.0698e-02,  6.5863e-02,  9.6843e-02],
                            [-7.7197e-02,  6.7764e-02,  8.8464e-02],
                            [-5.2054e-02,  9.6890e-02,  7.9019e-02]],
                  
                           [[ 1.1544e-03,  5.0823e-02, -3.6853e-02],
                            [-9.1936e-02,  2.6645e-02,  3.1425e-02],
                            [-6.8891e-02,  5.1123e-02, -9.0043e-02]],
                  
                           [[ 9.0718e-02,  1.0208e-01,  2.8699e-02],
                            [-6.6137e-02,  5.1300e-02,  1.7963e-02],
                            [ 2.8663e-02,  3.4643e-02,  8.0254e-02]],
                  
                           [[-4.5309e-02, -2.3711e-02,  2.8746e-02],
                            [ 1.1486e-02,  8.5000e-02, -5.5365e-02],
                            [-3.8387e-03,  1.9696e-02, -2.7996e-02]],
                  
                           [[ 7.1859e-02,  1.1530e-02, -9.7422e-02],
                            [-1.1420e-02, -4.7809e-02,  1.0243e-02],
                            [-1.2250e-02, -1.0456e-01, -1.9208e-02]],
                  
                           [[-1.0096e-02, -3.1083e-02,  9.6848e-02],
                            [-2.3000e-02,  6.7717e-02,  2.6112e-02],
                            [-8.8979e-02,  2.4770e-02,  8.7356e-02]],
                  
                           [[-6.8948e-02, -6.8134e-02,  1.0318e-01],
                            [ 8.4697e-02, -5.8807e-02,  6.3429e-02],
                            [-1.3485e-02, -1.0393e-01,  7.9198e-03]],
                  
                           [[ 3.4057e-02, -3.1619e-02,  3.6670e-02],
                            [-9.0136e-02,  7.3050e-02,  8.9865e-02],
                            [ 5.8130e-02,  1.7866e-02,  3.4716e-02]]],
                  
                  
                          [[[-7.6269e-02, -2.6339e-02, -1.0063e-02],
                            [-5.8659e-02, -7.7857e-02,  7.0900e-02],
                            [ 7.1535e-02, -9.5731e-02,  3.3542e-02]],
                  
                           [[ 4.2881e-02,  1.0014e-01,  6.0985e-02],
                            [ 9.6907e-02, -3.4510e-02,  7.3827e-02],
                            [ 8.5740e-02, -9.9541e-02, -8.4613e-02]],
                  
                           [[ 2.1335e-02,  5.7557e-02, -5.2369e-02],
                            [ 1.1609e-02, -1.5303e-04,  2.6680e-02],
                            [-5.6642e-02,  5.9455e-02,  7.0098e-02]],
                  
                           [[-7.3139e-02,  1.0211e-03,  2.9247e-04],
                            [ 3.3849e-02,  9.8198e-02,  3.0913e-02],
                            [-2.3951e-02,  9.4672e-02, -4.0112e-02]],
                  
                           [[-3.0608e-02,  7.1969e-03, -8.0270e-02],
                            [ 1.1470e-02, -7.1518e-02,  1.0838e-02],
                            [ 1.0099e-02,  1.4591e-02, -8.8891e-02]],
                  
                           [[-1.0012e-01,  4.8501e-02,  9.0399e-02],
                            [-9.3537e-02,  3.9043e-02, -7.7594e-02],
                            [ 6.6082e-03,  9.8068e-02,  7.9965e-02]],
                  
                           [[-7.7069e-02,  6.5203e-02,  5.5057e-02],
                            [-1.6168e-04,  1.0211e-01, -4.1866e-02],
                            [-2.4530e-02, -5.3275e-02,  1.5168e-02]],
                  
                           [[ 2.7911e-02,  8.3990e-03, -5.9307e-02],
                            [-4.7452e-02,  3.5855e-02, -9.2426e-02],
                            [-1.6416e-02, -2.3350e-03, -4.2708e-02]],
                  
                           [[ 3.8360e-02,  6.7940e-03,  7.4004e-02],
                            [-9.3616e-03, -6.6528e-02,  7.4477e-02],
                            [ 1.4720e-02, -3.0189e-02, -6.9476e-02]],
                  
                           [[ 2.4707e-02, -1.0053e-01,  2.7762e-02],
                            [ 5.2119e-02, -9.2465e-02, -6.9009e-02],
                            [-7.5781e-02,  8.8597e-02,  8.9611e-02]]],
                  
                  
                          [[[ 6.5987e-03,  9.8959e-02, -3.5239e-02],
                            [-1.0233e-01,  3.6819e-02,  3.7343e-02],
                            [ 1.0334e-01, -3.0510e-05,  8.0785e-02]],
                  
                           [[ 6.4612e-02,  7.6292e-02, -1.0460e-01],
                            [ 8.6800e-02, -8.9856e-02,  9.4501e-02],
                            [-4.3682e-03, -9.3415e-02,  2.9314e-02]],
                  
                           [[-2.1456e-02, -9.4678e-02, -3.8215e-02],
                            [ 1.0868e-02,  8.2098e-02, -3.2406e-02],
                            [ 6.2610e-02,  1.3200e-02,  3.5531e-03]],
                  
                           [[ 2.0170e-02, -6.9177e-02, -8.7616e-02],
                            [-3.3121e-02, -9.8226e-02, -4.9158e-02],
                            [ 4.8494e-03, -6.9424e-02, -4.3723e-02]],
                  
                           [[-1.8941e-02, -1.2144e-02, -5.8187e-02],
                            [ 5.0650e-03, -1.4795e-02,  3.0147e-02],
                            [ 4.7611e-03, -5.2638e-02, -3.6291e-02]],
                  
                           [[-1.2149e-03, -6.5774e-02,  8.2520e-03],
                            [-7.4425e-03,  4.0897e-02,  2.4947e-02],
                            [ 7.8887e-02, -3.4749e-03, -7.7887e-02]],
                  
                           [[ 4.7119e-02, -7.1240e-02, -1.4489e-02],
                            [-3.4132e-02, -3.9997e-02, -3.9000e-02],
                            [ 9.6863e-02,  6.0342e-02,  2.9213e-02]],
                  
                           [[ 9.8975e-02, -9.5524e-02,  1.7010e-02],
                            [ 6.7481e-02,  7.0022e-02, -8.3890e-02],
                            [ 3.7514e-02, -6.0050e-02, -4.1187e-03]],
                  
                           [[-2.1996e-02, -8.8013e-02, -1.0055e-01],
                            [-6.9349e-02,  4.7832e-02,  4.8218e-02],
                            [-9.1681e-02, -3.9586e-02,  1.7218e-03]],
                  
                           [[-9.1135e-02,  5.9393e-02,  9.5473e-02],
                            [ 1.8643e-02, -7.8321e-02,  2.4580e-02],
                            [ 3.8265e-02,  8.3468e-02, -5.6085e-02]]],
                  
                  
                          [[[-9.4437e-02,  4.6312e-02,  6.5624e-03],
                            [-3.4345e-02, -4.4169e-02, -5.4351e-02],
                            [ 8.5328e-02, -1.8187e-02,  7.6022e-02]],
                  
                           [[ 9.4094e-02,  1.3353e-02,  2.2454e-02],
                            [-7.1789e-03,  7.2397e-02, -9.4983e-02],
                            [ 4.1919e-02, -1.7174e-02,  4.8132e-02]],
                  
                           [[-4.6949e-04, -3.9029e-02, -1.1379e-02],
                            [ 5.6920e-02, -7.3210e-02, -6.6629e-02],
                            [-2.3611e-02, -3.8235e-02,  4.1409e-02]],
                  
                           [[ 7.0937e-02, -1.1289e-02,  9.9672e-02],
                            [-4.4042e-02, -5.9151e-02, -4.7191e-02],
                            [-7.2624e-02, -7.3885e-02, -9.3921e-02]],
                  
                           [[-9.3422e-02,  2.7512e-02,  6.4284e-02],
                            [ 9.8963e-02,  8.9787e-02, -6.0709e-03],
                            [ 2.0454e-02, -6.3068e-02,  4.0743e-02]],
                  
                           [[-1.0107e-01,  4.9719e-02,  1.9334e-02],
                            [ 3.2393e-02,  3.8595e-02, -4.8394e-02],
                            [ 9.0452e-02,  5.0307e-02,  6.9243e-02]],
                  
                           [[ 1.3922e-02,  6.6196e-02,  7.0941e-02],
                            [ 4.7775e-02,  8.0297e-02, -1.9119e-02],
                            [ 6.9310e-02,  2.4286e-02,  6.3424e-02]],
                  
                           [[ 1.0267e-01,  2.3869e-02, -3.9124e-02],
                            [-1.0488e-02,  2.9676e-02,  1.7773e-02],
                            [-2.8795e-02,  8.2590e-02,  6.3331e-02]],
                  
                           [[-6.5475e-02, -8.5889e-03, -1.0119e-02],
                            [-6.6063e-02,  1.5374e-02, -3.2360e-02],
                            [-5.4419e-02, -3.3894e-02, -3.7584e-02]],
                  
                           [[ 1.0084e-01,  4.0432e-02,  1.0373e-01],
                            [ 2.8903e-02,  2.3868e-02,  4.3333e-02],
                            [ 1.8092e-02, -8.2722e-02, -6.2334e-02]]],
                  
                  
                          [[[-2.5538e-02,  1.5846e-03,  3.9709e-02],
                            [ 4.0588e-02,  8.3623e-02,  2.1458e-02],
                            [-3.5975e-02, -7.9271e-02, -7.7203e-02]],
                  
                           [[-6.2965e-02,  3.1792e-02,  5.6950e-02],
                            [ 9.2224e-02, -3.3342e-02, -8.3150e-03],
                            [-3.1303e-02, -3.8517e-04,  3.3837e-02]],
                  
                           [[-2.3160e-03,  4.8799e-03,  1.3354e-02],
                            [ 3.9256e-02, -3.1981e-02, -6.2855e-02],
                            [ 2.4869e-02, -1.2481e-02, -4.7753e-02]],
                  
                           [[ 4.4268e-02,  9.5597e-04, -1.5333e-02],
                            [-5.1027e-02, -1.3868e-02, -8.9632e-02],
                            [ 2.3980e-02,  1.5818e-03,  6.3966e-02]],
                  
                           [[ 6.8063e-03,  8.4277e-03,  2.8715e-02],
                            [ 8.0210e-02, -4.9812e-02,  6.2930e-02],
                            [ 2.5779e-02, -7.0320e-02,  3.6702e-02]],
                  
                           [[-6.3217e-02, -3.3181e-02, -5.0245e-02],
                            [-7.1711e-02,  8.3017e-02, -9.4217e-02],
                            [ 5.2706e-02, -9.4870e-02, -1.2829e-02]],
                  
                           [[ 6.2868e-03,  7.4937e-02, -3.8147e-02],
                            [ 3.0340e-02,  1.6329e-02,  6.2021e-02],
                            [ 6.2667e-03,  3.9470e-02, -6.3677e-02]],
                  
                           [[-7.3250e-02,  9.3928e-02, -7.6808e-02],
                            [-1.7945e-02, -1.2742e-02,  1.0308e-01],
                            [-2.2780e-02, -8.0249e-02, -2.6721e-02]],
                  
                           [[ 5.4372e-02,  4.1773e-02,  8.7204e-02],
                            [-2.1579e-02,  4.9653e-02, -9.9194e-02],
                            [ 4.0787e-02,  4.8432e-02,  6.7998e-02]],
                  
                           [[-6.0446e-02, -2.8142e-02,  2.5502e-02],
                            [-7.4905e-02, -8.3851e-02, -1.0141e-01],
                            [ 5.8842e-03,  6.5458e-02,  2.7075e-02]]],
                  
                  
                          [[[ 6.4263e-03,  3.6727e-02, -6.6240e-02],
                            [ 1.1113e-02, -2.6186e-02, -5.2193e-02],
                            [ 9.0902e-02, -8.1550e-02,  1.5448e-02]],
                  
                           [[-9.2624e-02, -3.5762e-03, -4.6840e-02],
                            [ 3.4695e-02, -5.9191e-02,  6.7466e-02],
                            [-8.5536e-02,  6.3313e-02, -7.9181e-02]],
                  
                           [[ 5.6456e-02, -4.4384e-02, -2.4556e-04],
                            [-1.9238e-02,  6.8414e-02,  3.4546e-02],
                            [-9.2887e-02,  9.6914e-03, -7.2718e-02]],
                  
                           [[ 7.8800e-02,  1.7319e-02, -2.7109e-02],
                            [-5.3777e-02,  3.6485e-02, -6.3129e-02],
                            [ 4.9992e-02,  5.7519e-02,  6.4701e-02]],
                  
                           [[ 2.7537e-02, -9.2272e-02,  7.5823e-02],
                            [-3.2700e-02, -3.1163e-02, -1.1325e-02],
                            [ 7.7068e-02,  8.1052e-02,  1.6276e-02]],
                  
                           [[ 5.0296e-02, -9.8241e-02,  2.4900e-04],
                            [-9.3254e-02,  3.5876e-02, -7.5099e-02],
                            [-3.7568e-02,  7.3684e-02,  1.0074e-01]],
                  
                           [[-6.3286e-02, -5.8503e-02,  1.3055e-02],
                            [ 4.1437e-02, -1.7168e-02, -3.2918e-02],
                            [-6.9237e-02,  4.4997e-02,  1.0328e-01]],
                  
                           [[-5.1026e-02,  4.9718e-02,  5.1481e-02],
                            [ 8.4728e-02, -1.2001e-02,  3.3202e-03],
                            [ 7.7444e-02,  6.6631e-02,  1.0411e-01]],
                  
                           [[-3.0207e-02,  4.1709e-02,  7.3605e-02],
                            [-7.1553e-02,  2.0940e-02, -2.3586e-02],
                            [ 6.7760e-02, -4.7342e-02,  7.3933e-03]],
                  
                           [[ 6.3067e-02, -9.6567e-02, -8.9004e-02],
                            [-5.3989e-02,  6.7611e-02,  7.0680e-02],
                            [-7.1991e-02,  2.0100e-02, -5.5854e-02]]],
                  
                  
                          [[[-4.8926e-02,  9.0907e-02,  5.0914e-02],
                            [-2.8828e-02,  1.5516e-02,  2.0424e-02],
                            [ 2.4691e-02, -3.6079e-02, -6.2074e-02]],
                  
                           [[ 6.9788e-02,  1.4164e-02,  4.4119e-02],
                            [-3.9922e-02,  5.1057e-02,  7.6713e-02],
                            [ 6.4107e-02,  2.8660e-02,  1.0371e-01]],
                  
                           [[-2.3053e-04,  2.2441e-02,  1.0015e-01],
                            [ 1.0245e-01, -4.4506e-02,  9.4953e-02],
                            [ 3.8902e-02, -1.1799e-02,  9.2038e-02]],
                  
                           [[-5.4605e-02,  6.8490e-02,  1.0445e-01],
                            [-7.2701e-02, -6.2201e-02, -1.0445e-01],
                            [-1.8970e-02, -9.5733e-02, -3.5304e-02]],
                  
                           [[ 3.2002e-02,  7.4511e-02,  5.8717e-02],
                            [ 5.8511e-02,  4.3730e-02, -6.5378e-02],
                            [-8.3694e-02,  4.3696e-03,  1.0009e-01]],
                  
                           [[ 5.9351e-03, -9.0662e-03, -7.1545e-02],
                            [-5.2266e-02, -8.1256e-02,  8.4398e-02],
                            [-1.7174e-02, -9.3119e-02,  1.1308e-02]],
                  
                           [[ 7.6494e-03, -1.3023e-02,  3.7733e-02],
                            [ 5.6687e-02, -9.9128e-02, -8.0753e-02],
                            [-5.0639e-03, -9.7729e-02, -9.5750e-02]],
                  
                           [[ 9.3067e-02, -8.0174e-03, -5.2113e-02],
                            [-3.6157e-02, -8.2295e-02,  8.2258e-02],
                            [-2.2857e-02, -5.9265e-02, -7.9944e-02]],
                  
                           [[ 6.1611e-02, -1.4571e-02, -1.1074e-02],
                            [-2.7473e-02, -5.0883e-02,  1.8751e-02],
                            [ 8.1099e-02, -6.1093e-02,  5.0504e-03]],
                  
                           [[-8.0165e-02, -4.9426e-02,  9.2525e-02],
                            [ 1.1052e-03,  1.0154e-01, -1.8468e-02],
                            [-5.7453e-02, -6.2981e-02,  9.3426e-02]]],
                  
                  
                          [[[-8.1058e-02,  5.5318e-02,  2.6203e-02],
                            [ 3.1107e-02,  5.9476e-02, -2.7577e-02],
                            [ 6.5223e-02, -8.3982e-02, -3.7087e-02]],
                  
                           [[ 7.7164e-02,  3.1283e-02, -1.4038e-02],
                            [-2.4616e-02, -6.4364e-02,  6.4098e-02],
                            [-3.3520e-03, -3.5664e-03,  2.4929e-02]],
                  
                           [[ 7.7787e-02, -5.3778e-02, -3.6303e-02],
                            [ 7.1429e-02,  5.9532e-02, -5.1855e-02],
                            [-1.0428e-01,  1.9555e-02,  5.5434e-02]],
                  
                           [[ 2.5178e-02,  7.4768e-02, -8.3640e-02],
                            [ 5.3156e-02, -6.5531e-02,  5.9325e-02],
                            [ 7.8394e-02,  3.3385e-02,  8.5284e-02]],
                  
                           [[-6.9481e-02, -9.4275e-02, -1.0135e-01],
                            [ 6.6179e-02,  3.6926e-02, -7.7188e-02],
                            [ 5.1048e-02,  9.6177e-02, -1.0394e-01]],
                  
                           [[ 7.6466e-02,  1.6167e-02,  9.8053e-03],
                            [ 9.4847e-02,  9.5458e-02,  4.4414e-02],
                            [ 8.3288e-02,  4.3853e-02,  1.7176e-02]],
                  
                           [[-9.2656e-02,  1.9689e-02, -7.4993e-02],
                            [ 3.2452e-02,  1.8598e-02,  2.3681e-03],
                            [-7.2071e-02, -6.3899e-02,  7.7912e-02]],
                  
                           [[ 5.1336e-02,  5.5576e-02, -3.1410e-02],
                            [-1.8151e-02, -2.7014e-02,  7.2489e-02],
                            [-4.5504e-02,  6.6394e-02,  7.2679e-02]],
                  
                           [[-9.6403e-02,  6.4369e-04, -2.0076e-02],
                            [-5.8273e-02,  4.5507e-02, -1.2807e-02],
                            [ 9.2287e-02, -6.5976e-02,  4.8976e-02]],
                  
                           [[-8.9998e-02, -5.2833e-02,  7.1903e-03],
                            [ 8.3283e-02,  5.5521e-02, -8.6550e-02],
                            [ 1.1676e-02, -6.2138e-02,  4.5674e-03]]]], device='cuda:0')),
                 ('conv_block_2.0.bias',
                  tensor([-0.0878, -0.0309,  0.0723, -0.0967, -0.1005,  0.0192,  0.0144, -0.0193,
                           0.0920, -0.0635], device='cuda:0')),
                 ('conv_block_2.2.weight',
                  tensor([[[[-6.3992e-02, -7.8791e-02, -1.9619e-02],
                            [-2.6901e-02,  6.5222e-02, -5.9186e-03],
                            [ 3.3663e-02, -4.3804e-02,  8.5507e-02]],
                  
                           [[ 8.8862e-02, -9.4401e-02, -2.7090e-02],
                            [-8.9439e-02,  4.4781e-02, -9.2094e-02],
                            [-4.9839e-02,  1.0532e-01, -1.0066e-01]],
                  
                           [[ 7.7771e-02,  8.9049e-03,  8.4289e-02],
                            [-5.3494e-02,  6.9236e-02,  1.2718e-02],
                            [ 8.1073e-03,  7.1945e-02, -1.0019e-01]],
                  
                           [[-8.4902e-02,  1.0180e-01, -6.3298e-02],
                            [-7.5980e-02, -5.1539e-03, -3.3742e-02],
                            [-1.4421e-02, -7.0623e-02,  3.8034e-02]],
                  
                           [[-9.0703e-02,  8.5374e-03,  6.1510e-02],
                            [ 2.0253e-02,  1.4006e-02,  1.5418e-02],
                            [-3.0880e-02, -2.0080e-02, -4.4450e-02]],
                  
                           [[-7.1207e-02, -5.5810e-02,  1.0420e-01],
                            [-1.7641e-02,  3.6924e-02,  7.2896e-02],
                            [-8.2343e-03, -5.6707e-02, -7.1419e-02]],
                  
                           [[-3.8833e-02,  3.7624e-02, -8.8771e-02],
                            [-1.2870e-02,  4.0096e-02,  8.5999e-02],
                            [ 3.1721e-02,  2.0846e-02,  7.2162e-02]],
                  
                           [[ 4.8708e-02,  3.5661e-02, -3.2682e-02],
                            [-8.4528e-02, -2.2769e-02, -1.9117e-02],
                            [ 7.7410e-03, -1.1593e-02,  4.2616e-02]],
                  
                           [[ 7.0050e-02, -4.2735e-02, -1.0002e-01],
                            [-5.4081e-02, -5.0436e-02,  5.9750e-02],
                            [-6.7994e-02, -9.9145e-03, -2.2340e-02]],
                  
                           [[-6.3976e-02,  4.7780e-02, -4.3909e-02],
                            [-5.4531e-03, -7.4112e-02, -1.0632e-02],
                            [ 1.4977e-02, -4.2894e-03, -3.9386e-02]]],
                  
                  
                          [[[ 3.1315e-02, -2.7311e-02, -5.8439e-02],
                            [-7.7732e-02, -2.2329e-02, -9.9578e-02],
                            [ 8.7492e-02, -5.0357e-02, -4.3684e-02]],
                  
                           [[ 9.7439e-03,  2.7326e-02, -9.9393e-03],
                            [ 7.2313e-02, -6.1448e-02,  3.7777e-02],
                            [-2.3773e-04, -8.5747e-02, -4.0824e-02]],
                  
                           [[ 2.6825e-02,  2.0138e-02,  7.6647e-02],
                            [ 7.0518e-02, -5.7493e-02, -4.5013e-02],
                            [-2.2351e-02, -7.5517e-02, -2.8459e-02]],
                  
                           [[-8.6258e-02,  4.0092e-02,  7.4583e-02],
                            [ 8.3459e-03, -7.5460e-02, -7.9827e-02],
                            [-4.1036e-02,  3.0659e-02,  2.5711e-03]],
                  
                           [[ 1.9166e-02,  9.9346e-02,  4.8956e-02],
                            [ 2.2665e-02, -2.1327e-02,  4.9864e-02],
                            [ 3.8563e-02, -9.4879e-02, -6.2266e-02]],
                  
                           [[ 3.5381e-03,  3.9997e-02,  5.1282e-02],
                            [-6.2748e-02, -1.0458e-01, -5.4909e-03],
                            [-1.2050e-02,  3.0588e-02, -2.8988e-02]],
                  
                           [[ 8.0588e-02,  7.0333e-03,  7.6975e-02],
                            [-7.3398e-02,  4.2167e-02,  1.2560e-02],
                            [-5.2720e-02,  5.2256e-02, -1.0372e-01]],
                  
                           [[ 8.5220e-02,  8.4947e-03,  1.0178e-02],
                            [ 4.8746e-02,  8.7503e-03,  4.5184e-02],
                            [ 6.7063e-02, -8.2268e-02,  6.9735e-02]],
                  
                           [[-1.5784e-02, -2.4513e-02,  2.1217e-02],
                            [ 8.2446e-02, -5.7302e-02, -7.1039e-02],
                            [ 6.5418e-02, -4.9507e-02,  3.3937e-02]],
                  
                           [[-1.5530e-02,  2.9014e-02,  8.0439e-02],
                            [-5.3421e-02, -5.1151e-02,  5.1716e-02],
                            [ 5.7714e-03, -1.1601e-02, -9.2590e-02]]],
                  
                  
                          [[[ 8.9309e-02, -3.9919e-03, -1.9415e-02],
                            [-4.3269e-02, -2.0801e-02,  5.1233e-02],
                            [-2.4227e-03,  9.0147e-02, -6.0858e-03]],
                  
                           [[-1.5122e-02,  5.9498e-02, -2.7275e-03],
                            [-2.1039e-02,  3.5231e-02,  8.3129e-02],
                            [ 2.6305e-02,  7.3398e-02,  6.8309e-02]],
                  
                           [[ 2.9810e-02,  3.6650e-02,  3.4014e-02],
                            [ 1.0934e-02,  8.9675e-02,  9.7308e-02],
                            [ 3.7524e-02, -5.2640e-03,  9.4509e-02]],
                  
                           [[-8.2042e-02,  7.7453e-02,  5.5849e-02],
                            [ 6.7687e-02, -8.0992e-03, -7.8646e-02],
                            [ 7.5193e-02, -4.6091e-02,  2.7734e-02]],
                  
                           [[ 5.9719e-02, -9.8508e-02,  6.9954e-03],
                            [-3.7444e-02,  7.4815e-02, -6.7114e-02],
                            [ 6.4001e-02,  6.5730e-02,  5.8156e-02]],
                  
                           [[ 1.0119e-01,  1.5964e-02, -9.5541e-02],
                            [ 7.5248e-02,  9.6499e-03,  2.0918e-03],
                            [-1.0041e-01, -2.3691e-02, -5.1162e-02]],
                  
                           [[ 1.0324e-01,  7.5054e-02,  7.8634e-02],
                            [ 7.2188e-02, -6.5340e-02, -4.5270e-02],
                            [-4.1252e-02, -4.2257e-02,  8.2054e-02]],
                  
                           [[ 3.5815e-02,  8.4470e-02, -4.9309e-03],
                            [-9.3965e-02, -3.0582e-02,  7.4081e-02],
                            [ 6.4174e-02,  3.2632e-02, -3.0919e-02]],
                  
                           [[-9.8386e-02, -5.6639e-02,  5.4958e-02],
                            [-4.2518e-02,  5.0421e-02,  2.8781e-02],
                            [-4.0486e-02,  6.4202e-02, -3.3871e-02]],
                  
                           [[-3.5020e-03, -4.0152e-02, -9.9988e-02],
                            [ 1.6996e-02,  3.0460e-02, -5.3072e-02],
                            [ 6.4663e-02, -9.4558e-02, -1.0161e-01]]],
                  
                  
                          [[[-6.5106e-02, -3.6430e-02, -1.1707e-02],
                            [-2.0370e-02,  4.8108e-02, -9.2510e-02],
                            [ 1.5521e-02,  1.8254e-03,  2.7842e-02]],
                  
                           [[ 1.0479e-01,  6.4874e-02, -5.8366e-02],
                            [-8.6378e-02, -2.5520e-02, -5.2876e-02],
                            [ 3.6820e-02,  9.6628e-04,  8.4783e-02]],
                  
                           [[ 4.1405e-02, -1.9382e-02,  3.6229e-03],
                            [ 2.5244e-02, -1.3080e-02,  8.5058e-02],
                            [-8.2420e-02,  5.1377e-02, -6.7192e-02]],
                  
                           [[-9.2347e-02, -2.1640e-02,  5.1366e-02],
                            [ 7.4478e-02,  2.6452e-02, -9.1104e-03],
                            [-5.9092e-03, -4.2731e-02, -9.4592e-03]],
                  
                           [[-7.2831e-03,  8.9699e-02,  6.1690e-02],
                            [-8.4351e-02,  4.3604e-04, -6.4834e-02],
                            [-1.6733e-02, -8.3776e-02,  2.7402e-02]],
                  
                           [[-7.6008e-02,  1.0406e-01,  7.9605e-02],
                            [-7.2559e-02, -9.9239e-02,  4.1128e-03],
                            [-2.9425e-02,  3.0945e-02, -7.1353e-02]],
                  
                           [[ 4.3148e-02, -9.1047e-02, -5.5632e-02],
                            [-5.5414e-02,  5.1007e-02, -2.7597e-03],
                            [-1.0130e-01, -6.0201e-02, -4.8781e-02]],
                  
                           [[-9.7802e-02,  1.3497e-02,  3.7561e-02],
                            [-1.9340e-02, -4.1947e-02, -6.3926e-04],
                            [-8.3725e-02, -6.4184e-02, -2.4040e-03]],
                  
                           [[ 9.3643e-02, -3.2414e-02,  5.2247e-02],
                            [-4.1484e-02, -2.8060e-02, -1.0034e-01],
                            [ 8.7330e-02,  1.0264e-01, -2.2139e-03]],
                  
                           [[ 6.6974e-02,  8.6219e-02,  5.2359e-02],
                            [ 5.4288e-02, -1.0035e-01, -9.9050e-02],
                            [-8.0906e-02,  3.2970e-02, -9.1177e-02]]],
                  
                  
                          [[[-8.0464e-02, -5.1092e-02, -9.7154e-02],
                            [ 1.4203e-04,  1.5207e-02, -6.1686e-02],
                            [ 6.9018e-02, -4.0018e-02, -2.9676e-02]],
                  
                           [[ 8.0309e-02,  9.0499e-02, -1.2093e-02],
                            [-7.5671e-02, -5.2881e-02,  1.3423e-02],
                            [ 6.1790e-02,  5.2477e-02, -4.6547e-02]],
                  
                           [[-9.9650e-02, -9.2249e-02, -3.3537e-02],
                            [ 1.3223e-03, -4.7347e-02, -8.3348e-02],
                            [ 1.1109e-02, -8.3668e-02, -8.0946e-02]],
                  
                           [[-8.5692e-02, -2.8563e-02,  9.3104e-02],
                            [ 4.1207e-02, -1.2498e-02,  2.1694e-02],
                            [ 4.1975e-02,  6.1414e-04, -8.5020e-02]],
                  
                           [[-6.4944e-02, -7.1610e-02, -2.6766e-03],
                            [-9.6492e-02, -1.9166e-02, -3.8545e-02],
                            [ 1.0345e-01,  8.5679e-02,  6.1227e-02]],
                  
                           [[ 5.9116e-03, -3.4129e-02,  2.6887e-02],
                            [-7.2830e-02, -4.4957e-02, -2.1175e-02],
                            [-2.4766e-02, -9.9854e-02,  4.1903e-02]],
                  
                           [[ 8.6803e-02, -5.8141e-02,  2.8415e-02],
                            [-1.2225e-02, -3.8445e-03,  6.1443e-03],
                            [ 9.1346e-02,  1.4124e-02, -6.6690e-02]],
                  
                           [[-3.7917e-02,  5.1495e-02,  3.2893e-02],
                            [ 2.0487e-03, -1.3912e-02, -4.1012e-02],
                            [-3.7413e-02, -5.5602e-02,  1.7273e-02]],
                  
                           [[ 2.9603e-02,  8.0717e-02, -2.3813e-02],
                            [ 7.5461e-03,  6.8125e-02,  4.5852e-02],
                            [ 1.3544e-02,  3.2390e-02,  5.4714e-03]],
                  
                           [[-9.0419e-02,  4.0636e-03, -2.3040e-02],
                            [ 9.5123e-02,  9.5145e-02,  2.0912e-02],
                            [ 9.4215e-02, -5.4288e-02,  9.1619e-02]]],
                  
                  
                          [[[ 9.0756e-02, -4.0288e-03, -8.4592e-02],
                            [-3.4015e-02, -2.8189e-02,  1.7411e-03],
                            [-9.5569e-02,  1.9535e-02, -4.3839e-02]],
                  
                           [[-2.6989e-02, -5.4443e-02, -2.2255e-02],
                            [-9.7896e-02, -5.5885e-02,  9.7108e-03],
                            [ 6.9072e-02,  9.5790e-02, -7.9737e-02]],
                  
                           [[ 4.4264e-02, -5.9419e-02, -8.1498e-02],
                            [-4.6417e-03, -6.0468e-02, -9.0783e-02],
                            [-9.8509e-02, -7.0556e-02,  8.6619e-02]],
                  
                           [[ 5.8788e-02, -4.1726e-02, -7.0553e-02],
                            [-8.1085e-02, -6.2246e-02, -4.3376e-02],
                            [ 6.3308e-02,  3.4496e-02, -4.0622e-02]],
                  
                           [[ 7.2567e-02, -6.5484e-02, -8.5876e-02],
                            [ 2.3006e-02, -5.8123e-02,  2.9987e-02],
                            [ 8.9306e-02, -4.9849e-02, -7.3556e-02]],
                  
                           [[ 3.9676e-02, -9.5200e-02,  9.4044e-02],
                            [-4.9780e-02,  5.0961e-02, -8.3818e-02],
                            [-7.1348e-02,  1.1611e-02,  3.7463e-02]],
                  
                           [[ 8.1734e-02,  8.8158e-02, -6.0623e-03],
                            [-1.3552e-02,  1.7424e-02, -2.4486e-02],
                            [ 3.5882e-03, -9.9828e-02, -8.6531e-02]],
                  
                           [[ 7.2233e-02, -6.1597e-02,  8.3008e-02],
                            [ 1.1568e-02,  2.5676e-02,  9.5804e-02],
                            [-5.8628e-02, -1.6640e-02,  1.8675e-02]],
                  
                           [[ 3.6012e-02, -1.0259e-01,  3.7464e-02],
                            [-6.2163e-02,  1.3846e-02,  7.1315e-02],
                            [-1.0500e-02, -3.3346e-03, -7.8757e-03]],
                  
                           [[ 8.7962e-02,  5.9907e-02,  1.7727e-02],
                            [-6.3437e-02, -5.7241e-02,  8.3964e-02],
                            [ 7.5834e-02,  6.1033e-02, -8.2189e-02]]],
                  
                  
                          [[[ 8.2092e-02, -1.0076e-02,  7.7661e-02],
                            [ 9.1553e-02,  1.1554e-02, -4.3863e-02],
                            [ 9.9153e-02, -5.4931e-02,  6.8876e-02]],
                  
                           [[-1.0108e-01, -3.3153e-02, -9.1902e-02],
                            [-4.7284e-02,  4.4759e-02, -7.5529e-02],
                            [-9.1158e-02,  7.5371e-02,  5.6270e-02]],
                  
                           [[-1.1527e-03, -7.4309e-02, -2.7927e-02],
                            [-3.4129e-02,  6.5100e-02, -3.4478e-02],
                            [-3.0360e-02, -7.4720e-02, -4.9646e-02]],
                  
                           [[ 5.7074e-02,  6.7914e-02,  1.5315e-02],
                            [-3.9549e-02,  1.0124e-01,  2.0806e-02],
                            [-4.0688e-02, -3.6535e-02, -1.4752e-02]],
                  
                           [[ 4.9974e-02,  3.8555e-02,  7.6418e-02],
                            [-4.7494e-03,  8.7183e-02, -4.2816e-02],
                            [-4.8547e-02, -3.8927e-02, -9.8896e-02]],
                  
                           [[-6.9195e-02, -9.5382e-02, -6.2294e-03],
                            [ 9.9374e-04, -2.7358e-02, -7.2035e-02],
                            [ 9.5637e-02, -3.4926e-02,  5.0233e-02]],
                  
                           [[ 7.3408e-02, -6.9291e-02, -1.3179e-02],
                            [ 6.0923e-02,  1.0218e-01, -1.3299e-02],
                            [ 7.6382e-02, -8.2732e-02, -6.8489e-02]],
                  
                           [[ 8.6682e-02, -9.9801e-03,  1.0414e-01],
                            [ 7.6651e-03, -4.3714e-02,  1.0011e-01],
                            [ 9.2179e-02,  9.7826e-03, -6.3900e-02]],
                  
                           [[-4.5639e-03, -5.0693e-02,  7.6810e-02],
                            [ 4.8829e-03,  2.2191e-02,  6.3927e-02],
                            [ 3.4916e-02, -6.5803e-02,  8.7566e-02]],
                  
                           [[ 6.4758e-02, -6.5073e-02,  7.9700e-02],
                            [ 2.9905e-02, -2.0750e-02, -7.5385e-02],
                            [-1.7490e-02, -1.0335e-01,  6.0163e-02]]],
                  
                  
                          [[[ 7.6343e-02, -3.0347e-02,  9.7720e-02],
                            [-3.9032e-02,  1.8051e-02, -7.3459e-02],
                            [-4.4565e-03,  4.2610e-02,  4.5403e-02]],
                  
                           [[-3.5346e-03, -5.3154e-02,  7.3680e-02],
                            [ 6.9788e-02,  1.6916e-02, -4.8475e-02],
                            [ 2.2349e-02,  2.8186e-04,  9.6302e-02]],
                  
                           [[ 1.5621e-02,  8.1301e-03,  7.2057e-03],
                            [ 5.6079e-02, -1.3024e-03,  9.0351e-02],
                            [ 5.4917e-02, -7.9650e-02, -1.2063e-06]],
                  
                           [[-8.9472e-02, -8.0934e-02,  2.0480e-02],
                            [ 2.3687e-02, -9.2246e-03,  1.0019e-01],
                            [-5.6627e-02, -4.4176e-02, -1.6881e-02]],
                  
                           [[ 6.3911e-04, -8.9284e-03,  9.4909e-02],
                            [-4.4519e-02, -5.5137e-02,  9.0599e-03],
                            [ 7.9171e-02,  2.5019e-02,  5.6787e-02]],
                  
                           [[ 2.0406e-02,  8.9839e-02,  6.3311e-02],
                            [ 7.5428e-02, -1.4198e-02, -8.7268e-02],
                            [-5.0002e-02,  3.5910e-02,  7.3950e-02]],
                  
                           [[-4.1184e-02,  8.7218e-02,  1.5150e-02],
                            [ 4.1869e-04,  4.1093e-03, -1.8623e-02],
                            [ 9.8683e-02,  4.5784e-03,  6.4564e-02]],
                  
                           [[-8.8967e-02, -5.4309e-02,  1.1852e-02],
                            [ 8.4169e-02,  5.0184e-02,  2.0076e-02],
                            [-1.0414e-01,  1.9816e-03, -6.9581e-02]],
                  
                           [[-9.0006e-02,  1.4414e-02, -6.6693e-02],
                            [ 9.5674e-02, -5.7294e-02,  3.3970e-02],
                            [ 6.1871e-02, -8.1928e-02,  5.3946e-02]],
                  
                           [[-1.4114e-02,  5.4619e-02,  1.0201e-01],
                            [-4.4922e-02, -4.5653e-02,  8.3753e-02],
                            [ 1.1722e-02, -1.0513e-02,  7.9971e-02]]],
                  
                  
                          [[[-5.0928e-02, -5.2047e-03,  7.2403e-02],
                            [ 4.1195e-02, -6.8180e-02,  2.7398e-02],
                            [-8.0368e-02, -5.7245e-02,  6.7779e-02]],
                  
                           [[-2.8093e-02, -5.3691e-02,  7.4717e-03],
                            [ 2.5759e-02, -6.5524e-02, -7.1084e-02],
                            [-1.0209e-01,  2.7236e-02, -6.8013e-02]],
                  
                           [[ 8.0331e-03, -2.3576e-02, -6.8923e-02],
                            [-3.3636e-02, -8.1027e-02, -5.5797e-02],
                            [-3.2857e-03, -9.0116e-02, -9.2447e-02]],
                  
                           [[ 7.8958e-02,  9.9188e-03, -4.6618e-02],
                            [-3.5047e-03,  7.8168e-02, -8.7939e-02],
                            [-5.5886e-02, -7.6226e-02, -7.6634e-03]],
                  
                           [[-3.6274e-03, -8.2146e-02,  7.3163e-02],
                            [-8.0946e-02,  9.8414e-02, -7.2560e-02],
                            [-1.4446e-02,  1.9710e-02, -4.6852e-02]],
                  
                           [[ 9.6939e-02, -7.2673e-02, -5.8427e-03],
                            [-7.7398e-02,  2.9261e-02,  8.9871e-02],
                            [ 9.7776e-02,  1.2514e-02, -5.2773e-02]],
                  
                           [[ 1.0244e-01,  7.8667e-03,  7.1317e-02],
                            [-5.4751e-02, -4.8920e-02, -8.7504e-02],
                            [ 9.6990e-02,  1.7486e-02, -7.5704e-02]],
                  
                           [[ 9.0535e-03, -4.5211e-02,  5.2659e-03],
                            [ 3.4988e-02, -5.2308e-02,  1.8394e-02],
                            [-6.6553e-02,  2.0312e-02, -1.0178e-01]],
                  
                           [[ 1.6797e-02,  1.0473e-01,  9.7094e-02],
                            [ 3.8451e-02,  7.7563e-02,  1.0248e-01],
                            [ 2.9870e-02,  3.5156e-02,  1.3707e-02]],
                  
                           [[ 9.3322e-02,  9.0551e-02, -4.9570e-02],
                            [-4.3333e-03, -5.3110e-02,  3.7824e-02],
                            [-1.0214e-01,  3.7301e-02, -2.8929e-02]]],
                  
                  
                          [[[ 3.8227e-02,  3.2899e-02, -5.2454e-02],
                            [ 5.4687e-02,  4.4762e-02, -8.9602e-02],
                            [ 1.0517e-01,  9.0731e-02,  6.5584e-02]],
                  
                           [[-1.0699e-02,  3.7345e-02, -5.7028e-02],
                            [-3.5818e-02,  4.9749e-02,  4.6925e-02],
                            [ 4.1741e-02, -1.0053e-01,  8.7350e-02]],
                  
                           [[-4.4028e-02,  9.1223e-02,  8.6852e-02],
                            [ 3.9070e-02,  1.0502e-01,  6.0528e-02],
                            [ 6.1821e-02, -3.5794e-02,  9.7766e-02]],
                  
                           [[ 2.7627e-02,  6.2280e-02, -2.3834e-02],
                            [ 7.6340e-02,  9.3509e-02, -8.0770e-02],
                            [ 8.6415e-02, -6.9664e-02, -7.2571e-02]],
                  
                           [[-8.8089e-02,  3.0459e-02, -7.9144e-02],
                            [-3.9680e-02, -5.2988e-02,  2.8172e-02],
                            [-1.0349e-01, -4.8324e-02,  7.7112e-04]],
                  
                           [[ 9.4660e-03, -4.7605e-02,  3.7764e-02],
                            [-6.9544e-02, -8.9270e-02, -1.4986e-02],
                            [-5.6989e-02,  6.6443e-02, -7.2049e-02]],
                  
                           [[-8.8494e-03,  4.3782e-02, -9.2311e-02],
                            [ 8.1599e-02, -4.7895e-02, -2.8684e-02],
                            [-6.4480e-02, -3.9279e-02, -4.0645e-02]],
                  
                           [[-9.3801e-02,  3.6019e-02, -3.3768e-04],
                            [ 1.0311e-01,  7.1117e-02,  9.1699e-02],
                            [ 3.1014e-02,  5.5388e-02,  9.8704e-02]],
                  
                           [[ 8.6545e-02, -8.0996e-02, -2.3636e-02],
                            [-1.0166e-01,  3.9877e-03, -3.7229e-02],
                            [ 9.1486e-02,  1.6666e-02,  1.1601e-03]],
                  
                           [[-7.6248e-02, -8.2718e-02,  1.6594e-02],
                            [-5.2376e-02, -4.8409e-02,  7.3938e-02],
                            [-5.4952e-02, -4.6918e-02,  8.0934e-02]]]], device='cuda:0')),
                 ('conv_block_2.2.bias',
                  tensor([ 0.0412, -0.0599,  0.0319,  0.0531, -0.0936,  0.0197,  0.0241, -0.0041,
                           0.1011, -0.0697], device='cuda:0')),
                 ('classifier.1.weight',
                  tensor([[ 0.0245, -0.0240, -0.0387,  ...,  0.0094, -0.0015, -0.0225],
                          [ 0.0228,  0.0067, -0.0439,  ..., -0.0302,  0.0368,  0.0293],
                          [ 0.0303,  0.0347, -0.0211,  ...,  0.0207, -0.0423, -0.0240],
                          ...,
                          [-0.0359, -0.0343,  0.0166,  ...,  0.0324,  0.0113, -0.0143],
                          [-0.0294, -0.0316,  0.0251,  ..., -0.0056,  0.0300, -0.0396],
                          [-0.0246, -0.0035, -0.0046,  ..., -0.0146, -0.0358,  0.0175]],
                         device='cuda:0')),
                 ('classifier.1.bias',
                  tensor([ 0.0320, -0.0445,  0.0246, -0.0357, -0.0442,  0.0156, -0.0010, -0.0277,
                           0.0404,  0.0037], device='cuda:0'))])

</div>

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
