
from engine.base_client.upload import BaseUploader
from typing import List
from dataset_reader.base_reader import Record

# just a stub, don't really do anything
class VortexUploader(BaseUploader):
    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        pass

    @classmethod
    def upload_batch(cls, batch: List[Record]):
        pass

    @classmethod
    def post_upload(cls, distance):
        return {}

    @classmethod
    def delete_client(cls):
        pass

