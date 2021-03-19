# SFE
Salient Feature Extractor for Adversarial Defense on Deep Neural Networks


Requirements
Python 3.5 
Tensorflow-gpu-1.3 
Tflearn-0.3.2. 

Folder images are examples of benign images and their adversarial examples in MNIST. The former can be 
correctly classified while the latter may lead to misclassification. 
In model folder is well-trained CNN1 model with acc=99.79%.

Running the model

We can use the GAN.py script to train the model to gain SF and TF for adversarial defense and detetion
on MNIST. Advdetector is trained in detector.py 
Correct labels will be obtained when SF of adv is input to the model for classification. 


## Abstract：
Recent years have witnessed unprecedented success achieved by deep learning models in the field of computer vision. However, the vulnerability of them towards carefully crafted adversarial examples has also attracted increasing attention of researchers. Motivated by the observation that the existence of adversarial examples is due to the non-robust feature learned from the original dataset by models, we propose the concepts of salient feature (SF) and trivial feature (TF). The former represents the class-related feature while the latter is usually adopted to mislead the model. We extract these two features with coupled GAN model and put forward a novel detection and defense method named salient feature extractor (SFE) to defend against adversarial attacks. Concretely, detection is realized by separating and comparing the difference between SF and TF of the input while correct labels are obtained by re-identification of SF, so as to reach the purpose of defense. Extensive experiments are carried out on MNIST, CIFAR-10 and ImageNet datasets where SFE shows the state-of-the-art results in effectiveness and efficiency when compared with baselines. Furthermore, we provide interpretable understanding of the defense and detection process.
