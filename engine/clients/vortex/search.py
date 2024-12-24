
import numpy as np
import requests

from typing import List, Tuple
from dataset_reader.base_reader import Query
from engine.base_client.search import BaseSearcher


import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    @classmethod
    def search_many(cls, query: List[Query], top: int, threads: int) -> Tuple[List[List[Tuple[int, float]]], List[float]]:
        reses = [None] * len(query)
        perfs = [0.0] * len(query)
        def search_single(index_query):

            """Perform search for a single query."""
            index, queryi = index_query

            try:
                start = time.perf_counter()
                
                data = np.array(queryi.vector).astype(np.float32).tobytes()
                req = cls.session.post(cls.url,data=data)
                if not req.ok:
                    print(f"Error: {req.text}")
                    raise Exception("req not ok")
                end = time.perf_counter()
                return index, end-start, [(int(x),0.0) for x in req.text.split("@")]
                
            except Exception as e:
                print(f"Error while searching for query {queryi}: {e}")
                raise

        # Use ThreadPoolExecutor for threading
        with ThreadPoolExecutor(max_workers=threads) as executor:  # Adjust max_workers as needed
            future_to_query = {executor.submit(search_single, (i, queryi)): i for i, queryi in enumerate(query)}

            for future in as_completed(future_to_query):
                try:
                    index, perf, result = future.result()  # Get index and result
                    reses[index] = result  # Place result in the correct position
                    perfs[index] = perf  # Place performance in the correct position
                except Exception as e:
                    print(f"Query failed with error: {e}")
                    raise

        return reses, perfs
