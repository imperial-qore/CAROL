[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/DRAGON/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FDRAGON&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/DRAGON/workflows/AIoT-Benchmarks/badge.svg)](https://github.com/imperial-qore/DRAGON/actions)
![Docker pulls AIoTBench](https://img.shields.io/docker/pulls/shreshthtuli/aiotbench?label=docker%20pulls%3AAIoTBench)

# DRAGON

CAROL: Confidence Aware Resilience Model for Edge Federations

Why is broker resilience crucial in edge federations? This is because if a worker fails, a broker can do its job (i.e. act as a worker) or allocate the same job to another worker. So fault remediation steps are possible and do not have a very high cost. However, if a broker fails, all tasks coming to that broker fail. The worker nodes can not be used. This makes broker resilience far more important. 

Explain what are brokers and workers. There is a tradeoff to the number of brokers we need in the system. We assume that gateway devices send to the closest broker, breaking ties uniformly at random. Now if brokers are too many, workers are too less impacting performance of the system. If brokers are too less, we have less single points of failures and possibly brokers become bottlenecks. So we can increase or decrease number of brokers but both need to be considered. 

Regular optimization techniques that use neural networks as a surrogate model claim that this approach is better because of goal-directed search (by using gradients compared to other search strategies. However, in discrete domains, there is an approximation that the surrogate surface would be smooth, i.e., the closest discrete point to the optimum would be optimum in the discrete space. However, this is not always true and can dive rise to non-optimal solutions.

Another problem with GOBI is that we have no way to find out the confidence of the surrogate surface, i.e. the approximation of the real metrics. This leads us to either perform uncertainty based optimization (GOSH) or add other parameters like topology etc to improve performance (HUNTER). Another problem arising from this is that we do not know when to fine-tune the model, so we need to do this at each interval which might not be the best decision considering variable and spiky loads can cause contention in constrained edge nodes. However, for adaptive systems, confidence is important to make sure we fine-tune only when needed. Thus instead of going from Graph -> NN -> Metrics, we need to go to Graph + Metrics -> NN -> confidence. This is similar to a discriminator network that predicts the probability of true data. We only need normal execution traces for this. Another advantage this GON provides is that we can now train using random samples to make sure that for unseen settings the confidence is lower, so that we can fine-tune the model. We can use POT to find the confidence thresholds below which we train the model with latest info on the new graph topology till the confidence becomes normal. 

So now that we have a GON model and we know when to fine-tune it for a new topology, we can now run fault-tolerance steps. Our GON model takes as inputs, the graph topology and metrics that are initialized randomly. Now, at each step, we start from the previous topology (starting topology set by federation manager) and use second-order gradient optimization to find the optimal metrics such that the metrics are as expected. The converged GON output is the confidence score. We run tabu search on the topology using the neighbours found by the various node-shifts.  

For experiments, we need only normal execution traces with diverse topologies to train the model. We need a fault model for test time.

## Figures


## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
