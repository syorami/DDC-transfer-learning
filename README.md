# DDC-transfer-learning
A simple implementation of Deep Domain Confusion: Maximizing for Domain Invariance. 
The project contains *Pytorch* code for fine-tuning *Alexnet* as well as *DDCnet*  implemented according to the original paper which adds an adaptation layer into the Alexnet. 
The *office31* dataset used in the paper is also used in this implementation to test the performance of fine-tuning *Alexnet* and *DDCnet* with additional linear *MMD* loss. What’s more, 
    
# Run the work
* Run command `python alextnet_finetune.py` to fine-tune a pretrained *Alexnet* on *office31* dataset with *full-training*.
* Run command `python DDC.py` to fine-tune a pretrained *Alexnet* on *office31* dataset with *full-training*.

# Experiment Results
Here we have to note that *full-training* protocol, which is taking all the samples from one domain as the source or target domain, and *dowm-sample* protocol, which is choosing 20 or 8 samples per category to use as the domain data, are quite different data preparation methods. 

| Methods | Results (amazon to webcame) |
| :------: | :------: |
| fine-tuning Alexnet (full-training) in *Pytorch* | Around 51% |
| DDC ( pretrained Alexnet with adaptation layer and MMD loss) in *Pytorch* | Around 56% |

# Future work
- [ ] Write data loader using  *down-sample* protocol mentioned in the paper instead of using *full-training* protocol.
- [ ] Considering trying a tensorflow version to see if frameworks can have a difference on final experiment results.
    
# Reference
Tzeng E, Hoffman J, Zhang N, et al. Deep domain confusion: Maximizing for domain invariance[J]. arXiv preprint arXiv:1412.3474, 2014.
