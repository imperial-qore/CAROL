[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/CAROL/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCAROL&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/DRAGON/workflows/AIoT-Benchmarks/badge.svg)](https://github.com/imperial-qore/CAROL/actions)
![Docker pulls AIoTBench](https://img.shields.io/docker/pulls/shreshthtuli/aiotbench?label=docker%20pulls%3AAIoTBench)

# CAROL

### Importance of Broker Resilience

Why is broker resilience crucial in edge federations? If a worker fails, a broker can do its job (i.e. act as a worker) or allocate the same job to another worker. So fault remediation steps are possible and do not have a very high cost. However, if a broker fails, all tasks coming to that broker fail. The worker nodes can not be used. This makes broker resilience far more important. 

There is a tradeoff to the number of brokers we need in the system. We assume that gateway devices send to the closest broker, breaking ties uniformly at random. If there are too many brokers, we have too less brokers, impacting performance of the system. If there are too less brokers, we have low number of single points of failures and possibly brokers can become bottlenecks. So we can increase or decrease number of brokers but both need to be considered. 

### Importance of Confidence-Aware Training

Regular optimization techniques that use neural networks as a surrogate model claim that this approach is better because of goal-directed search (by using gradients compared to other search strategies. However, in discrete domains, there is an approximation that the surrogate surface would be smooth, i.e., the closest discrete point to the optimum would be optimum in the discrete space. However, this is not always true and can give rise to non-optimal solutions.

Another problem with GOBI is that we have no way to find out the confidence of the surrogate surface, i.e. the approximation of the real metrics. This leads us to either perform uncertainty based optimization ([GOSH](https://arxiv.org/abs/2112.08916)) or add other parameters such as system topology to improve performance ([HUNTER](https://arxiv.org/abs/2110.05529)). A problem arising from this is that we do not know when to fine-tune the model, so we need to do this at each interval, which might not be the best decision as variable and spiky loads can cause contention in constrained edge nodes. However, for adaptive systems, confidence is important to make sure we fine-tune only when needed. Thus instead of going from input graph to QoS metrics (Graph -> Metrics), we need to go from input and metrics to a confidence score (Graph + Metrics -> confidence score). This is similar to a discriminator network that predicts the probability of true data. We only need normal execution traces for this. We use a GON model here as we can now train using random samples and make sure that for unseen settings the confidence is lower (and facilitate decision making on when to fine-tune the model). We can use POT to find the confidence thresholds below which we train the model with latest info on the new graph topology till the confidence score crosses above the threshold value. 

### CAROL Approach

So now that we have a GON model and we know when to fine-tune it for a new topology, we can now run fault-tolerance steps. Our GON model takes as inputs, the graph topology and metrics that are initialized randomly. Now, at each step, we start from the previous topology (starting topology set by federation manager) and use second-order gradient optimization to find the optimal metrics such that the metrics are as expected. The converged GON output is the confidence score. We run tabu search on the topology using the neighbours found by the various node-shifts. 

For experiments, we need only normal execution traces with diverse topologies to train the model. We need a fault model for test time.

## Quick Test
Clone repo.
```console
git clone https://github.com/imperial-qore/CAROL.git
cd CAROL/
```
Install dependencies.
```console
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
export PATH=$PATH:~/.local/bin
```
Change line 115 in `main.py` to use one of the implemented fault-tolerance techniques: `CAROLRecovery`, `ECLBRecovery`, `DYVERSERecovery`, `ELBSRecovery`, `LBOSRecovery`, `FRASRecovery`, `TopoMADRecovery` or `StepGANRecovery` and run the code using the following command.
```console
python3 main.py
````

## External Links
| Items | Contents | 
| --- | --- |
| **Pre-print** | (In progress) |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship |

## Cite this work
Our work is accepted in IEEE Conference on Computer Communications (INFOCOM) 2022. Cite our work using the bibtex entry below.
```bibtex
@inproceedings{tuli2022carol,
  title={{CAROL: Confidence-Aware Resilience Model for Edge Federations}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  booktitle={IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)},
  year={2022},
  organization={IEEE}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
