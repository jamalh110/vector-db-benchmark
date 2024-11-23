from typing import List, Tuple

from weaviate import WeaviateClient
from weaviate.classes.config import Reconfigure
from weaviate.classes.query import MetadataQuery
from weaviate.collections import Collection
from weaviate.connect import ConnectionParams

from dataset_reader.base_reader import Query
from engine.base_client.search import BaseSearcher
from engine.clients.weaviate.config import WEAVIATE_CLASS_NAME, WEAVIATE_DEFAULT_PORT
from engine.clients.weaviate.parser import WeaviateConditionParser

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class WeaviateSearcher(BaseSearcher):
    search_params = {}
    parser = WeaviateConditionParser()
    collection: Collection
    client: WeaviateClient

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        url = f"http://{host}:{connection_params.get('port', WEAVIATE_DEFAULT_PORT)}"
        client = WeaviateClient(
            ConnectionParams.from_url(url, 50051), skip_init_checks=True
        )
        client.connect()
        cls.collection = client.collections.get(
            WEAVIATE_CLASS_NAME, skip_argument_validation=True
        )
        cls.search_params = search_params
        cls.client = client

    @classmethod
    def search_one(cls, query: Query, top: int) -> List[Tuple[int, float]]:
        res = cls.collection.query.near_vector(
            near_vector=query.vector,
            filters=cls.parser.parse(query.meta_conditions),
            limit=top,
            return_metadata=MetadataQuery(distance=True),
            return_properties=[],
        )
        return [(hit.uuid.int, hit.metadata.distance) for hit in res.objects]
    
    @classmethod
    def search_many(cls, query: List[Query], top: int, threads: int) -> Tuple[List[List[Tuple[int, float]]], List[float]]:
        reses = [None] * len(query)
        perfs = [0.0] * len(query)
        def search_single(index_query):

            """Perform search for a single query."""
            index, queryi = index_query
            try:
                start = time.perf_counter()
                
                res = cls.collection.query.near_vector(
                    near_vector=queryi.vector,
                    filters=cls.parser.parse(queryi.meta_conditions),
                    limit=top,
                    return_metadata=MetadataQuery(distance=True),
                    return_properties=[],
                )

                end = time.perf_counter()
                return index, end-start, [(hit.uuid.int, hit.metadata.distance) for hit in res.objects]
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
    
    def setup_search(self):
        self.collection.config.update(
            vector_index_config=Reconfigure.VectorIndex.hnsw(
                ef=self.search_params["config"]["ef"]
            )
        )

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            cls.client.close()
