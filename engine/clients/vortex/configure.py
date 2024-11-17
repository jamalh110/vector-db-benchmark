
from engine.base_client.configure import BaseConfigurator
from benchmark.dataset import Dataset

# just a stub, don't really do anything
class VortexConfigurator(BaseConfigurator):
    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        print("Vortex configurator created")

    def clean(self):
        pass

    def recreate(self, dataset: Dataset, collection_params):
        pass


    def delete_client(self):
        pass

