<h1 align="center">
  <br/>
    Reliable ReLU Toolbox (RReLU) To Enhance Resilience of DNNs 
  </br>
</h1>
<p align="center">
<a href="#background">Background</a> •
<a href="#usage">Usage</a> •
<a href="#code">Code</a> •
<a href="#citation">Citation</a> •
</p>

## Background
The Reliable ReLU Toolbox (RReLU) is a powerful reliability tool designed to enhance the resiliency of deep neural networks (DNNs) by generating reliable ReLU activation functions. Implemented for the popular PyTorch deep learning platform, RReLU allows users to find a clipped ReLU activation function using various methods. This tool is highly versatile for dependability and reliability research, with applications ranging from resiliency analysis of classification networks to training resilient models and improving DNN interpretability.

RReLU includes all state-of-the-art activation restriction methods. These methods offer several advantages: they do not require retraining the entire model, avoid the complexity of fault-aware training, and are non-intrusive, meaning they do not necessitate any changes to an accelerator. RReLU serves as the research code accompanying the paper [insert paper title], and it includes implementations of the following algorithms:

* **ProAct** (the proposed algorithm) ([code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/proact.py)).
* **FitAct** ([paper](https://arxiv.org/pdf/2112.13544) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/fitact.py)).
* **FtClipAct** ([paper](https://arxiv.org/pdf/1912.00941) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/ftclip.py)).
* **Ranger** ([paper](https://arxiv.org/pdf/2003.13874) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/ranger.py)).

## Usage

Download on PyPI [here]().

### Installing

**From Pip**

Install using `pip install rrelu`

**From Source**

Download this repository into your project folder.

### Importing

Import the entire package:

```python
import rrelu
```

Import a specific module:

```python
from rrelu.search_bound import proact_bounds 
```

### Testing
-- running and evaluating algorithms: 
```python
torchpack dist-run -np 1 python rrelu.search.py --dataset "dataset name (CIFAR10, CIFAR100)" --data_path "path to the dataset" --model "name of the model" --init_from "pretrained file path" \
                      --name_relu_bound "name of bounded relu" --name_serach_bound "name of the search algorithm" --bounds_type "type of thresholds" --bitflip "value representaiton"
```


## Code

### Structure

The main source code of framework is held in `src/rrelu`, which carries `search_bounds`, `relu_bounds` , `extended pytorchfi` and other  implementations.


## Citation

View the [published paper](). If you use or reference rrelu, please cite:

```

```
