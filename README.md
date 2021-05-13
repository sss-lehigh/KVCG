# KVCG

This is the implementation of KVCG used for the paper
"KVCG: A Heterogeneous Key-Value Store for Skewed Workloads"
by dePaul Miller, Jacob Nelson, Ahmed Hassan, and Roberto Palmieri.

## Abstract

We present KVCG, a novel heterogeneous key-value store whose primary 
objective is to serve client requests targeting frequently accessed 
(hot) keys at sub-millisecond latency and requests targeting less 
frequently accessed (cold) keys with high throughput. To accomplish 
this goal, KVCG deploys an architecture where requests on hot keys 
are routed to a software cache operated by CPU threads, while the 
remainder are offloaded to a data repository optimized for execution 
on modern GPU devices. Cold/hot partitioning is done at runtime through 
a model trained with the incoming workload. Against a state-of-the-art 
competitor, we obtainup to 34x improvement in latency.

## Building

Make sure to boostrap vcpkg and install tbb, gtest, boost-system, and boost-property-tree.

Then use cmake with -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake

## Startup File

KVCG loads in a JSON file on startup to determine what to run.

- threads
    - number of threads for hot cache
- gpus
    - number of gpus
- model
    - file for the model to load in
- train
    - boolean to decide whether requests should be sampled to train the model
- size
    - size of the map to create
- batchSize
    - size of batch
- cache
    - bool whether to use cache

## Main API

- updateModel
    - tells the server to update the model
- batch
    - sends a batch of up to batchSize key, value, and request type

## Correctness Notes

Relies on CPU having TSO memory model.