import multiprocessing as mp
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import Collection, connections
import time
from dataset_reader.base_reader import Query
from engine.base_client.search import BaseSearcher
from engine.clients.milvus.config import (
    DISTANCE_MAPPING,
    MILVUS_COLLECTION_NAME,
    MILVUS_DEFAULT_ALIAS,
    MILVUS_DEFAULT_PORT,
)
from engine.clients.milvus.parser import MilvusConditionParser


class MilvusSearcher(BaseSearcher):
    search_params = {}
    client: connections = None
    collection: Collection = None
    distance: str = None
    parser = MilvusConditionParser()

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        cls.client = connections.connect(
            alias=MILVUS_DEFAULT_ALIAS,
            host=host,
            port=str(connection_params.get("port", MILVUS_DEFAULT_PORT)),
            **connection_params
        )
        cls.collection = Collection(MILVUS_COLLECTION_NAME, using=MILVUS_DEFAULT_ALIAS)
        cls.search_params = search_params
        cls.distance = DISTANCE_MAPPING[distance]

    @classmethod
    def get_mp_start_method(cls):
        return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    @classmethod
    def search_one(cls, query: Query, top: int) -> List[Tuple[int, float]]:
        param = {"metric_type": cls.distance, "params": cls.search_params["config"]}
        try:
            res = cls.collection.search(
                data=[query.vector],
                anns_field="vector",
                param=param,
                limit=top,
                expr=cls.parser.parse(query.meta_conditions),
            )
        except Exception as e:
            import ipdb

            ipdb.set_trace()
            print("param: ", param)

            raise e

        return list(zip(res[0].ids, res[0].distances))

    @classmethod
    def search_many_old(cls, query: List[Query], top: int) -> List[List[Tuple[int, float]]]:
        param = {"metric_type": cls.distance, "params": cls.search_params["config"]}
        reses = []
        try:
            for queryi in query:
                res = cls.collection.search(
                    data=[queryi.vector],
                    anns_field="vector",
                    param=param,
                    limit=top,
                    expr=cls.parser.parse(queryi.meta_conditions),
                )
                reses.append(list(zip(res[0].ids, res[0].distances)))
        except Exception as e:
            import ipdb

            ipdb.set_trace()
            print("param: ", param)

            raise e

        return reses
    
    @classmethod
    def search_many(cls, query: List[Query], top: int, threads: int) -> Tuple[List[List[Tuple[int, float]]], List[float]]:
        param = {"metric_type": cls.distance, "params": cls.search_params["config"]}
        reses = [None] * len(query)
        perfs = [0.0] * len(query)
        def search_single(index_query):

            """Perform search for a single query."""
            index, queryi = index_query
            try:
                start = time.perf_counter()
                res = cls.collection.search(
                    data=[queryi.vector],
                    anns_field="vector",
                    param=param,
                    limit=top,
                    expr=cls.parser.parse(queryi.meta_conditions),
                )
                end = time.perf_counter()
                return index, end-start, list(zip(res[0].ids, res[0].distances))
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