import pandas as pd
import os
import targetBN

# Each function for structure learning
# Search method: Hill-Climbing
def Hill_Climbing(dataset: pd.DataFrame):
    # from pgmpy.estimators import ExhaustiveSearch
    from pgmpy.estimators import HillClimbSearch
    from pgmpy.estimators import BDeuScore, K2Score, BicScore

    # bdeu = BDeuScore(dataset, equivalent_sample_size=5)
    # k2 = K2Score(dataset)
    # bic = BicScore(dataset)

    hc = HillClimbSearch(dataset, scoring_method=BicScore(dataset))
    best_model = hc.estimate()
    # print(best_model.edges())
    return best_model.edges()


# Search method: Constraint-based
def Constraint_based(dataset: pd.DataFrame):
    from pgmpy.estimators import ConstraintBasedEstimator

    est = ConstraintBasedEstimator(dataset)

    # Construct dag
    skel, seperating_sets = est.estimate_skeleton(significance_level=0.01)
    print("Undirected edges:", skel.edges())

    pdag = est.skeleton_to_pdag(skel, seperating_sets)
    print("PDAG edges:", pdag.edges())

    model = est.pdag_to_dag(pdag)
    print("DAG edges:", model.edges())

    # print(est.estimate(significance_level=0.01).edges())
    print(type(model))


# Search method: Hybrid structure learning
def Hybrid(dataset: pd.DataFrame):
    from pgmpy.estimators import MmhcEstimator
    from pgmpy.estimators import BDeuScore

    mmhc = MmhcEstimator(dataset)
    skeleton = mmhc.mmpc()
    print("Part 1) Skeleton: ", skeleton.edges())

    # use hill climb search to orient the edges:
    hc = HillClimbSearch(dataset, scoring_method=BDeuScore(dataset))
    model = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges())
    print("Part 2) Model:    ", model.edges())
    print(type(model.edges()))

# File path
access_rights = 0o755
data_dir = "dataset"
data_fname = data_dir + "/Covid-19-dataset.pxl"
model_dir = "/model"
model_name = ""

# Generate a dataset or open a stored one
# size of dataset
sample_size = 30

if __name__== "__main__":
    if(not os.path.exists(data_fname)):
        if(not os.path.exists(data_dir)):
            try:
                os.mkdir(data_dir, access_rights)
            except OSError:
                print("Permission denied: creating directory=>", data_dir)
            else:
                print("Successfully create directory for dataset!")
        generator = targetBN.TargetBayesNet()
        dataset = generator.getDataset(sample_size)
        dataset.to_pickle(data_fname)
    else:
        dataset = pd.read_pickle(data_fname)
        # Once the sampling size changes, recreate the dataset againg
        if(len(dataset.index) != sample_size):
            generator = targetBN.TargetBayesNet()
            dataset = generator.getDataset(sample_size)
            dataset.to_pickle(data_fname)
            
    # print(dataset)

    edges = Hill_Climbing(dataset)

    # Starting with defining the network structure
    # Creating the model as well as the structure (arcs)
    from pgmpy.models import BayesianModel

    # create a new BN
    covid_model = BayesianModel(edges)

    # Estimating the CPTs from the given dataset
    covid_model.fit(dataset)

    # Checking if the cpds are valid for the model.
    print("Checking if CPDs are valid for model: ", covid_model.check_model())

    # Probability reasoning-----------------------------------
    from pgmpy.inference import VariableElimination
    covid_infer = VariableElimination(covid_model)

    # Computing the probability of bronc given smoke.
    q = covid_infer.query(variables=["Covid"], evidence={"Fever": 1})
    print(q)

    q = covid_infer.query(variables=["Covid"], evidence={"Fever": 1, "Difficulty_in_Breathing": 1})
    print(q)

    # Given the result of cancer, find P(+f|+c)
    q = covid_infer.query(variables=["Fever"], evidence={"Covid": 1})
    print(q)

    q = covid_infer.query(variables=["Covid"], evidence={})
    print(q)

    q = covid_infer.query(variables=["Covid"], evidence={"Fever": 0, "Difficulty_in_Breathing": 0, "Tiredness": 1, "Dry_Cough": 0})
    print(q)
