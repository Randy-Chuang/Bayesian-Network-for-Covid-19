# -*- coding: utf-8 -*-
# @Filename : targetBN.py
# @Description: Generating training data from a target bayesian network created by ourselves. 
# @Date : 2020-June
# @Project: Early detection of Covid-19 using BN (AI Term project)
# @AUTHOR : Randy Chuang

import process

# Cause and symptoms in the BN.
Contact = "Contact"
Covid = "Covid"
None_Symptom = "None_Symptom"
Diarrhea = "Diarrhea"
Fever = "Fever"
Dry_Cough = "Dry_Cough"
Tiredness = "Tiredness"
Loss_of_taste_or_smell = "Loss_of_taste_or_smell"
Difficulty_in_Breathing = "Difficulty_in_Breathing"
Sore_Throat = "Sore_Throat"

nodes = [Contact, Covid, None_Symptom, Diarrhea, Fever, Dry_Cough, Tiredness, Loss_of_taste_or_smell, Difficulty_in_Breathing, Sore_Throat]

class TargetBayesNet:
    def __init__(self, model_path=""):
        """
        Class for generating dataset from pre-defined Bayesian Network.
        """
        # Starting with defining the network structure
        # Creating the model as well as the structure (arcs)
        import numpy as np
        from pgmpy.models import BayesianModel
        
        # Define the directed connection of BN
        edges_list = [(Contact, Covid), 
                        (Covid, None_Symptom), 
                        (Covid, Dry_Cough), 
                        (Covid, Fever), 
                        (Covid, Loss_of_taste_or_smell), 
                        (Covid, Diarrhea), 
                        (Covid, Difficulty_in_Breathing), 
                        (Covid, Sore_Throat), 
                        (None_Symptom, Dry_Cough), 
                        (None_Symptom, Fever), 
                        (None_Symptom, Loss_of_taste_or_smell), 
                        (None_Symptom, Diarrhea), 
                        (None_Symptom, Difficulty_in_Breathing), 
                        (None_Symptom, Sore_Throat), 
                        (Dry_Cough, Difficulty_in_Breathing), 
                        (Dry_Cough, Sore_Throat), 
                        (Dry_Cough, Tiredness), 
                        (Fever, Tiredness)]

        # Initialize BN with the connection of directed edgse
        self.__covid_model = BayesianModel(edges_list)

        # Defining the parameters.
        # Specifying the CPD for each node 
        # http://pgmpy.org/factors.html#module-pgmpy.factors.discrete.CPD
        # TabularCPD values: for example(3 variables): this node and 2 parents
        # First-dimension: this node 
        # second-dimension: cartesian-product of the values from parents

        from pgmpy.factors.discrete import TabularCPD

        # Having contact with confirmed patients 
        cpd_Contact = TabularCPD(variable=Contact, variable_card=2,
                                values=[[0.9], [0.1]])
        
        # Conditional probability of containing Covid-19 (assumption: measuring from the patients who asked for test in hospital)
        cpd_Covid = TabularCPD(variable=Covid, variable_card=2,
                                values=[[0.7, 0.4], 
                                        [0.3, 0.6]], 
                            evidence=[Contact], evidence_card=[2])

        cpd_None = TabularCPD(variable=None_Symptom, variable_card=2,
                            values=[[0.1, 0.7], 
                                    [0.9, 0.3]], 
                            evidence=[Covid], evidence_card=[2])

        cpd_Dry = TabularCPD(variable=Dry_Cough, variable_card=2,
                            values=[[0.95, 0.98, 0.3, 0.99], 
                                    [0.05, 0.02, 0.7, 0.01]], 
                            evidence=[Covid, None_Symptom], evidence_card=[2, 2])

        cpd_Fever = TabularCPD(variable=Fever, variable_card=2,
                            values=[[0.95, 0.99, 0.15, 0.99], 
                                    [0.05, 0.01, 0.85, 0.01]], 
                            evidence=[Covid, None_Symptom], evidence_card=[2, 2])

        cpd_Loss = TabularCPD(variable=Loss_of_taste_or_smell, variable_card=2,
                            values=[[0.999, 0.999, 0.7, 0.99], 
                                    [0.001, 0.001, 0.3, 0.01]], 
                            evidence=[Covid, None_Symptom], evidence_card=[2, 2])

        cpd_Diarrhea = TabularCPD(variable=Diarrhea, variable_card=2,
                            values=[[0.85, 0.99, 0.75, 0.99], 
                                    [0.15, 0.01, 0.25, 0.01]], 
                            evidence=[Covid, None_Symptom], evidence_card=[2, 2])

        cpd_Diff = TabularCPD(variable=Difficulty_in_Breathing, variable_card=2,
                            values=[[0.999, 0.95, 0.999, 0.999, 0.8, 0.6, 0.999, 0.999], 
                                    [0.001, 0.05, 0.001, 0.001, 0.2, 0.4, 0.001, 0.001]], 
                            evidence=[Covid, None_Symptom, Dry_Cough], evidence_card=[2, 2, 2])

        cpd_Sore = TabularCPD(variable=Sore_Throat, variable_card=2,
                            values=[[0.95, 0.7, 0.999, 0.999, 0.8, 0.3, 0.999, 0.999], 
                                    [0.05, 0.3, 0.001, 0.001, 0.2, 0.7, 0.001, 0.001]], 
                            evidence=[Covid, None_Symptom, Dry_Cough], evidence_card=[2, 2, 2])

        cpd_Tiredness = TabularCPD(variable=Tiredness, variable_card=2,
                            values=[[0.95, 0.35, 0.5, 0.05], 
                                    [0.05, 0.65, 0.5, 0.95]],
                            evidence=[Dry_Cough, Fever], evidence_card=[2, 2])

        # cpd_cancer = TabularCPD(variable='', variable_card=2,
        #                         values=[[0.03, 0.05, 0.001, 0.02],
        #                                 [0.97, 0.95, 0.999, 0.98]],
        #                         evidence=['', ''],
        #                         evidence_card=[2, 2])

        # Associating the parameters with the model structure.
        self.__covid_model.add_cpds(cpd_Contact, cpd_Covid, cpd_None, cpd_Dry, cpd_Fever, cpd_Loss, cpd_Diarrhea, cpd_Diff, cpd_Sore, cpd_Tiredness)

        # Checking if the cpds are valid for the model.
        print("Bayesian Network generated successfully or not: ", self.__covid_model.check_model())
        
        graph_file = ""
        model_file = ""
        if(model_path != ""):
            graph_file = model_path+"/"
            model_file = model_path+"/"
        graph_file+="targetBN"
        model_file+="targetBN.bif"
        
        # process.saveGraphToPDF(graph_file, list(self.__covid_model.edges()), True)
        process.saveModel(self.__covid_model, model_file)

        # Doing some simple queries on the network
        # Check if there is active trail between the nodes
        # self.__covid_model.is_active_trail('', '')
        # self.__covid_model.is_active_trail('', '', observed=[])
        # self.__covid_model.local_independencies('')

        # Checking all the independencies
        # print(self.__covid_model.get_independencies())

    
    def getDataset(self, size = 1000, return_type = 'DataFrame'):
        """
        Method: retrun a set of samples generated from Bayesian Network. (Simply using forward-sampling)

        Parameters
        ----------
        size: size of the dataset to be generated (default: 1000)

        return_type: return type of dataset (default: panda.DataFrame)

        """
        # For more info, see: likelihood_weighted, rejection or Gibb sampling
        from pgmpy.sampling import BayesianModelSampling

        inference = BayesianModelSampling(self.__covid_model)
        dataset = inference.forward_sample(size=size, return_type=return_type)

        return dataset        

 
if __name__== "__main__":
	generator = TargetBayesNet()
	dataset = generator.getDataset(3000)
	print(dataset)