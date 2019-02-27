# HW2 - Jython Implementation

To adapt the `ANN_*.py` files to your dataset:
- Change the `INPUT_LAYER` variable to reflect the number of features.
- Replace `bank_test.csv` and `bank_train.csv` in each file with your dataset.

Run with

```
jython ANN_back.py
```

- Add support for parameters if you got too much time
- Push helpful changes to this repo

---------------


Please add anything you find useful for coding up hw2 using scikit-learn or Python implementation thank you.


https://github.com/skylergrammer/SimulatedAnnealing -- scikit learn simulated annealing
https://github.com/rsteca/sklearn-deap -- scikit learn genetic algorithm

MIMIC implementation in Python:
https://github.com/mjs2600/mimicry

Isbell's Paper on MIMIC:
https://www.cc.gatech.edu/~isbell/papers/isbell-mimic-nips-1997.pdf


---------------------------------------------------
Winnie's Approach to Installing Dependencies on ubuntu
1) Install Java
2) Install Apache Ant
3) Install Jypthon 
- https://www.jython.org/archive/22/installation.html
- Command line to install Jython: 
  sudo java -jar jython_installer-2.2.1.jar
  - this is the home path for the jython interpreter: ~/jython2.2.1/jython   [~/jython2.5.2/bin/jython for stack overflow example]
  - set PATH as this  home path: PATH=$HOME/jython2.5.2/:$PATH [ PATH=$HOME/jython2.5.2/bin:$PATH for stack overflow example]
 4) Git clone Abigail
 
