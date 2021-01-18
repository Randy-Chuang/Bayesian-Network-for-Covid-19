# -*- coding: utf-8 -*-
# @Filename : analysis.py
# @Description: Analyzing the chosen model with the dataset.
# @Date : 2020-June
# @Project: Early detection of Covid-19 using BN (AI Term project)
# @AUTHOR : Randy Chuang
import pandas as pd
import os
import targetBN
import numpy as np
import matplotlib.pyplot as plt

# File path
access_rights = 0o755
data_dir = "dataset"
data_fname = data_dir + "/Covid-19-validation.pxl"
model_dir = "model"
model_name = model_dir + "/Learned_model"

# Generate a dataset or open a stored one
# size of dataset
sample_size = 1000

if __name__== "__main__":
    if(not os.path.exists(model_dir)):
        try:
            os.mkdir(model_dir, access_rights)
        except OSError:
            print("Permission denied: creating directory=>", model_dir)
        else:
            print("Successfully create directory for storing model!")

    if(not os.path.exists(data_fname)):
        if(not os.path.exists(data_dir)):
            try:
                os.mkdir(data_dir, access_rights)
            except OSError:
                print("Permission denied: creating directory=>", data_dir)
            else:
                print("Successfully create directory for storing dataset!")

        generator = targetBN.TargetBayesNet(model_path=model_dir)
        dataset = generator.getDataset(sample_size)
        dataset.to_pickle(data_fname)
    else:
        dataset = pd.read_pickle(data_fname)
        # Once the sampling size changes, recreate the dataset againg
        if(len(dataset.index) != sample_size):
            generator = targetBN.TargetBayesNet(model_path=model_dir)
            dataset = generator.getDataset(sample_size)
            dataset.to_pickle(data_fname)
            
    # print(dataset)
    
    Model_files = [f for f in os.listdir("model") if os.path.isfile(os.path.join("model", f)) and f.endswith(".bif")]
    
    if(len(Model_files)):
        print("If you find there is only a target BN model, please try to perform structure learning first (training.py)")
        print("BN model: ", Model_files)
        selection = int(input("Please choose a model to perform analysis on it (index from 1):"))
        
        if(1<=selection and selection<=len(Model_files)):
            model_path = os.path.join("model", Model_files[selection-1])
            from pgmpy.readwrite import BIFReader
            reader = BIFReader(model_path)
            model = reader.get_model()
            # Checking if the cpds are valid for the model.
            print("Checking if CPDs are valid for model: ", model.check_model())
            # Probability reasoning with the samples from dataset
            # The result will be designated into two class (actually confirmed or healthy) with two colors in graph
            from pgmpy.inference import VariableElimination
            covid_infer = VariableElimination(model)
            variables_name = ["Covid"]
            evidences_name = list(dataset)
            evidences_name.remove("Covid")
            variables_list = dataset["Covid"].values.tolist()
            evidences_list = dataset.drop(columns=["Covid"]).values.tolist()
            
            import time
            start_time = time.time()

            confirmed = []
            healthy = []
            for indicator, evidences in zip(variables_list, evidences_list):
                evidence_dict = {key: str(evidence) for key, evidence in zip(evidences_name, evidences)}
                q = covid_infer.query(variables=variables_name, evidence=evidence_dict, show_progress=False)
                
                if(indicator):
                    confirmed.append(q.values[1])
                else:
                    healthy.append(q.values[1])

            print("--- %s seconds ---" % (time.time() - start_time))

            
            import numpy as np
            import seaborn as sns
            import matplotlib.pyplot as plt

            confirmed = np.array(confirmed)
            healthy = np.array(healthy)


            sns.set(style="white", palette="muted", color_codes=True)

            # Set up the matplotlib figure
            f, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

            axes[0].title.set_text('Covid-19 patients: '+str(len(confirmed))+' cases')
            axes[1].title.set_text('Healthy people: '+str(len(healthy))+' cases')
            axes[0].set(xlabel='Probability of infected by Covid-19 \n(given evidences)', ylabel='Number of cases')
            axes[1].set(xlabel='Probability of infected by Covid-19 \n(given evidences)', ylabel='Number of cases')

            # Covid-19 patients
            sns.distplot(confirmed, bins=20, kde=False, rug=True, color="r", ax=axes[0])

            # Healthy people
            sns.distplot(healthy, bins=20, kde=False, rug=True, color="b", ax=axes[1])

            plt.tight_layout()

            plt.show()


        else:
            print("Selection out of scope!")
    else:
        print("There is no model for analysis!")

   