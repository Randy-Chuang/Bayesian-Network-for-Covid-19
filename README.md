Term Project - Early detection of COVID-19 with Bayesian Network
====

## About The Project

- Background
    As COVID-19 continues to plague areas of the world, it is important to examine methods to reduce the spread of the virus which has currently been the leading cause of death.
    
    Even though some countries have different ways to detect COVID-19 such as swab test or blood test, it still takes a long time to wait the result comes up.
    
- Problem formulation
    - Problem:
        We want to find out a solution for early detection of COVID-19 or distinguishing high-risk and low-risk patients while we've got the symptoms from the patients. Thus, we would be able to prioritize the patients and give the proper treatments to them.
        
    - Framing Problem:
        We can frame this situation as a probability reasoning with **Bayesian Network** in AI.
        
        For example: <img src="svgs/36a8c50c9d2d4cab8dfee65f7ae75c74.svg?invert_in_darkmode" align=middle width=199.8648894pt height=28.5845208pt/> 

        Once we know the symptoms from a person, and then calculate the probability of being infected by COVID-19.

- Methodology
    To perform probability reasoning, we separate the implementation into 3 steps:
    1. Getting or generating dataset which contains cases of healthy people and patients with their symptoms.
    2. Structure learning: How to learn the structure of **Bayesian Network** and represent the real causality of the symptoms from the given dataset. 
    3. Analysis: Validating the learned **Bayesian Network** with a validation dataset.

    Since we hadn’t found a real dataset of COVID-19 patients with their symptoms, we decided to generate the dataset ourselves. We surveyed the symptoms of COVID-19 and created a simple **Bayesian Network** (Target Network) with reasonable connection and conditional probability distribution. With the target network, we would be able to generate a dataset containing patients and healthy people in it.

    With the dataset, we would be able to perform structure learning and try to construct the network connection and the appropriate conditional probability distribution.

- For detailed explanation, see our [project report](https://drive.google.com/file/d/1y2CA0RqRJhCPxCF0tXmQg_BOIbpzY_f8/view?usp=sharing).



### Platform and Software

Platform: **Ubuntu/Linux**

Softwares:
- [Python 3](https://www.python.org/downloads/)
- Python modules: 
    - [pgmpy](https://pypi.org/project/pgmpy/)
    - [matplotlib](https://matplotlib.org/)
    - Other dependencies.
- [GNU Make](https://www.gnu.org/software/make/)

## Getting Started

Simply using makefile target to build the project.

- Building the whole project.
    ```shell
    $ make
    ```
    or
    ```shell
    $ make all
    ```
- Building Python 3 virtual environment and installing required dependencies.
    ```shell
    $ make build
    ```
- Generating target **Bayesian Network** and dataset. Performing structure learning on the given dataset.
    ```shell
    $ make training
    ```
- Running analysis for the learned **Bayesian Network**.
    ```shell
    $ make analysis
    ```

## References

- Python module [pgmpy](https://pypi.org/project/pgmpy/) and its [documentation](http://pgmpy.org/).
- [Kaggle](https://www.kaggle.com/): [COVID-19 Symptoms Checker
](https://www.kaggle.com/iamhungundji/covid19-symptoms-checker)
- [GNU Make](https://www.gnu.org/software/make/)
