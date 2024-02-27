Node Compromising Detection to Mitigate Poisoning Attacks in IoT Networks
 
In this paper, we propose a framework, namely NoComP for Node Compromising detection, to defend against poisoning attacks by detecting compromised nodes and delete their readings from the collected data. NoComP prevents datasets to be mixed poisonous collected sensed data. To this end, we use as machine learning algorithm the neural network to detect compromised nodes. This algorithm offer significant advantages in terms of efficiency and accuracy in detecting anomalies. We carry out some experiments to evaluate NoComP and compare it with two existing proposals in the literature. The accuracy and efficiency results shows that NoComP outperforms the existing ones and improves the robustness against poisoning attacks.  
The present study exploits a dataset from IoT devices that record temperature readings. These IoT devices containing this data are installed both outside and inside an anonymous room. (https://www.openml.org/search?type=data&status=active&id=43351&sort=runs). As technical details, the original model dataset has 5 columns whose labels are: id: unique identifiers for each reading, room\_id/id: identifier of the room in which the device was installed (inside and/or outside), noted\_date: date and time of the reading, temp: temperature readings and out/in: whether the reading was taken from a device installed inside or outside the room. In total, this data set contains 97606 lines. 
Finally, we present here in two files the source code used for our experiments for detecting and removing compromised nodes from the network. The third file is the one containing the dataset we have used.
