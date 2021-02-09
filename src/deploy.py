
from azureml.core import Workspace, Datastore, Dataset, Run, Model
import pandas as pd
import os

# get the run 
# get the current run
run = Run.get_context()
ws = run.experiment.workspace

# get the registered model
model = Model(ws, name="sklearn-model")

# define the service name
service_name = 'my-sklearn-service'
service = Model.deploy(ws, service_name, [model], overwrite=True)
