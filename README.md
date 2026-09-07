# FALL
This is the Github Repository containing code for "Fermat Active Laplace Learning For Semi-supervised Hyperspectral Image Classification".

# Running the code

To reproduce all results from the FALL paper, install the required dependencies and run the following lines of code for each respective dataset: 

```powershell
python run_FALL.py --dataset salinasA  
python run_FALL.py --dataset paviaU_crop   
 ```

To run one (or more) algorithms, use the following flag:

```powershell
python run_FALL.py --algorithms fall
python run_FALL.py --algorithms a-fall
```

For the A-FALL algorithm, ALOO is the default method used for p-learning. To use ELOO p-learning, run the following command:

```powershell
python run_FALL.py --dataset paviaU_crop --algorithms a-fall --p-selection-method eloo
```

# Contact Information 
Email: vutichart.buranasiri@tufts.edu

(c) Vutichart Buranasiri, 2026
