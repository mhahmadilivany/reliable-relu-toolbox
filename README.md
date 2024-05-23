<h1 align="center">
  <br/>
    Reliable ReLU Tollbox (RRT) To Enhance Resilience of DNNs 
  </br>
</h1>
<p align="center">
<a href="#background">Background</a> •
<a href="#usage">Usage</a> •
<a href="#code">Code</a> •
<a href="#contributors">Contributors</a> •
<a href="#citation">Citation</a> •
<a href="#license">License</a>
</p>

## Background
RRT is the reliabilty tool that enchance the resiliency of deep neural networks (DNNs) with generating a reliable relu actibvation function, implemented for the popular PyTorch deep learning platform.
RRT enables users to find a clipped ReLU activation function based on the different methods.  It is extremely versatile for dependability and reliability research, with applications including resiliency analysis of classification networks, training resilient models, and for DNN interpertability. RRT contain all state-of-the-art activation restriction methods. These methods do not require retraining the entire model and do not suffer from the complexity of fault-aware training. In addition, they are non-intrusive in the sense that they do not require any changes to an accelerator. RRT is the research code that accompanies the paper ..... It contains implementation of the following algorithms:

* **ProAct** (the proposed algorithm) ([code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/proact.py)).
* **FitAct** ([paper](https://arxiv.org/pdf/2112.13544) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/fitact.py)).
* **FtClipAct** ([paper](https://arxiv.org/pdf/1912.00941) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/ftclip.py)).
* **Ranger** ([paper](https://arxiv.org/pdf/2003.13874) and [code](https://github.com/hamidmousavi0/reliable-relu-toolbox/blob/master/src/search_bound/ranger.py)).

