# bootstrapping neural_process

Codes for NeurIPS 2020 submission bootstrapping neural process

* Requirements
```
torch >= 1.4.0
pyyaml
attrdict
tqdm
```

* Before running the experiments, modify the ```ROOT``` variable in ```regression/utils/paths.py```  and ```bayesian_optimization/utils/path.py``` to your own ROOT path.

* Running 1D regression experiments
  * Go to ```regression``` folder.
  * Train a model by
  ```
  python gp.py --model model --expid expid --gpu gpu 
  ```
  where ```model``` can be ```cnp, np, bnp, canp, anp, banp```.  
  * To test your model, 
  ```
  python gp.py --model model --expid expid --mode eval
  ```
  * If you want to test OOD settings, for instance using periodic kernel,
  ```
  python gp.py --model model --expid expid --mode eval --eval_kernel periodic
  ```
  For t-noise setting,
  ```
  python gp.py --model model --expid expid --mode eval --t_noise -1
  ```
  
* Running EMNIST/CelebA experiments
  * Almost same as 1D regression experiments, except for EMNIST run ```emnist.py``` and for CelebA run ```celeba.py```.
  * ```emnist.py``` takes ```--class_range```, which is by default ```0 10```. To test on OOD setting with unseen classes, set ```--class_range 10 47```.
  
* Running Prey-predator experiments
  * Similar but run with ```lotka_volterra.py```.
  * To test on OOD (Real data), give ```--hare_lynx``` option.
  
 * Bayesian optimization
  * Train all the models using ```regression/gp.py```.
  * Go to ```bayesian_optimization``` folder.
  * Execute ```test_all.sh```.

  
