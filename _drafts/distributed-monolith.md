---
layout: post
title:  The Distributed Monolith
date:   2017-08-07 09:05:00+02
categories:
    - Architecture
---

More and more companies are adopting an architectural style they call "microservices" architecture. They started splitting their software into separately deployed instances, communicating with each other over the network. These kinds of architectures promise a couple of benefits. Namely independent deployability, the possiblity to choose the appropriate technology for a given problem and codebases small teams can handle. Yet a lot of those companies are not happy with the results they get.

### Symptoms
Instead of the benefits, they see behaviours like these:

* Small changes in one part of the system lead to failures in another part. 
* Environments are hard to keep "green". They break multiple times a day, sometimes with the reason hard to determine.
* Configuration of the environment is complicated and time-consuming.
* Changes in the system become harder and harder over time.
* Business transactions involve lots of different systems, leading to a need of some sort of distributed transactions.

Those of you who have some experience might have noticed that these problems were previously observed in large monolithic codebases.

### The Problem
While these issues might have lots of different sources, they might actually point to the same basic issue: while services where separated and distributed from each other they still are tightly coupled. Welcome the Distributed Monolith. If a system is in this state, it combines the bad properties from a microservices architecture with the issues from a monolithic architecture.

### Decoupling - Not Just Separation
In a Distributed Monotlith architecture, remote calls were introduced without assessing the benefits. Often these extracted interfaces are badly designed because these are *"just internal interfaces"* and therefore don't need to adhere to general API best practices.

But don't worry if you already work with a Distributed Monolith. You still can start refactoring your architecture: 

* **Introduce *real* interfaces** You decided to separate a service from another. This means you need to have stable, backwards compatible or versioned interfaces between them, which allow you to evolve your services without breaking their clients. 
* **Value your business transactions** If a lot of different systems are involved and you needed to introduce some sort of distributed transactions, it might be reasonable to analyze your transactions and try to write to only one system at a time.
* **Reconsider** While working with the system, you might have noticed parts of the system where you think the separation does not provide any value. In this cases it might be reasonable to reintegrate the two services again. 


### Conclusion

Don't fall into the trap of the Distributed Monolith! A microservices architecture can provide a lot of value, but you need to make sure you address the [downsides][TradeOffs] as well. There is no free lunch.

[TradeOffs]:     https://martinfowler.com/articles/microservice-trade-offs.html