*************
TO RUN T-NID
*************

DEPENDENCIES:
- numpy
- pytorch
- scipy
- sklearn

COMMANDS:
$ cd T-NID
$ python3 run.py --num_trials 1

OUTPUT:
Results directory with AUC scores for full power sweep of aggregations and representative samples
Results/14/output.txt includes LaTeX code to generate tables for all aggregations and representative samples

****************
TO RUN TAYLORCAM
****************

DEPENDENCIES:
- numpy
- cv2
- pickle
- pil
- torchvision
- pytorch 1.5

COMMANDS:
$ cd TaylorCAM

To generate Sort-Of-CLEVR dataset:
$ cd Data
$ python3 sort_of_clevr.py
$ cd ..

To train RN:
$ python3 RelationalReasoning.py
--OR--
To train IRN:
$ python3 RelationalReasoning.py --pre-relational

To visualize RN interactions:
$ python3 explain_relation_network.py
--OR--
To visualize IRN interactions:
$ python3 explain_relation_network.py --pre-relational

OUTPUT:
Results directory with interaction visuals and a stats file with test accuracy and AMIS for one random test batch