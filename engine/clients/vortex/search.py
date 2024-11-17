
import numpy as np
import requests

from typing import List, Tuple
from dataset_reader.base_reader import Query
from engine.base_client.search import BaseSearcher

class VortexSearcher(BaseSearcher):
    session = None
    url = None

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        cls.session = requests.Session()
        cls.url = connection_params["url"]

    @classmethod
    def search_one(cls, query: Query, top) -> List[Tuple[int, float]]:
        data = np.array(query.vector).astype(np.float32).tobytes()
        req = cls.session.post(cls.url,data=data)
        if req.ok:
            return [(int(x),0.0) for x in req.text.split("@")] # only the ID matters for the benchmark
        else:
            print(f"Error: {req.text}")
            return []

    @classmethod
    def delete_client(cls):
        cls.session.close()

