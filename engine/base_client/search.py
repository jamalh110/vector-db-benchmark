import functools
import time
from multiprocessing import get_context
from typing import Iterable, List, Optional, Tuple
from itertools import islice
import numpy as np
import tqdm
from datetime import datetime

from dataset_reader.base_reader import Query

DEFAULT_TOP = 10
def sleep_until_next_target():
    time.sleep(5)
    while True:
        now = datetime.now()
        target_seconds = [0, 10, 20, 30, 40, 50]
        # Find the next target second
        next_target = min(s for s in target_seconds if s > now.second) if any(s > now.second for s in target_seconds) else target_seconds[0]
        
        # Calculate the time to sleep
        time_to_sleep = (next_target - now.second) % 60  - now.microsecond / 1_000_000
        time.sleep(time_to_sleep)
        
        now = datetime.now()  # Update time after sleeping
        print(now.second, now.microsecond)
        if now.second in target_seconds:
            break
    
    print(f"Hit target second ({now.second}) at: {now}")

class BaseSearcher:
    MP_CONTEXT = None

    def __init__(self, host, connection_params, search_params):
        self.host = host
        self.connection_params = connection_params
        self.search_params = search_params
        
    def batch_iterable(self, iterable, batch_size):
        """Helper function to split an iterable into chunks of size batch_size."""
        iterator = iter(iterable)
        while chunk := list(islice(iterator, batch_size)):
            yield chunk

    @classmethod
    def init_client(
        cls, host: str, distance, connection_params: dict, search_params: dict
    ):
        raise NotImplementedError()

    @classmethod
    def get_mp_start_method(cls):
        return None

    @classmethod
    def search_one(cls, query: Query, top: Optional[int]) -> List[Tuple[int, float]]:
        raise NotImplementedError()
    
    @classmethod
    def search_many(cls, query: List[Query], top: Optional[int]) -> Tuple[List[List[Tuple[int, float]]], List[float]]:
        raise NotImplementedError()

    @classmethod
    def _search_one(cls, query: Query, top: Optional[int] = None):
        if top is None:
            top = (
                len(query.expected_result)
                if query.expected_result is not None and len(query.expected_result) > 0
                else DEFAULT_TOP
            )

        start = time.perf_counter()
        search_res = cls.search_one(query, top)
        end = time.perf_counter()

        precision = 1.0
        if query.expected_result:
            ids = set(x[0] for x in search_res)
            precision = len(ids.intersection(query.expected_result[:top])) / top

        return precision, end - start

    @classmethod
    def _search_many(cls, query: List[Query], top: Optional[int] = None, threads: int = 1):
        if top is None:
            top = (
                len(query[0].expected_result)
                if query[0].expected_result is not None and len(query[0].expected_result) > 0
                else DEFAULT_TOP
            )

        start = time.perf_counter()
        search_res, perfs = cls.search_many(query, top, threads)
        end = time.perf_counter()

        precision = 0.0
        for queryi, search_resi in zip(query, search_res):
            if queryi.expected_result:
                ids = set(x[0] for x in search_resi)
                precision += len(ids.intersection(queryi.expected_result[:top])) / top
        precision /= len(query)

        return precision, perfs

    def search_all(
        self,
        distance,
        queries: Iterable[Query],
    ):
        parallel = self.search_params.get("parallel", 1)
        batch = self.search_params.get("batch", 1)
        threads = self.search_params.get("threads", 1)
        oldway = self.search_params.get("oldway", False)
        print("oldway", oldway)
        top = self.search_params.get("top", None)
        
        # setup_search may require initialized client
        self.init_client(
            self.host, distance, self.connection_params, self.search_params
        )
        self.setup_search()

        search_one = functools.partial(self.__class__._search_one, top=top)
        search_many = functools.partial(self.__class__._search_many, top=top, threads=threads)

        if parallel == 1:
            start = time.perf_counter()
            precisions, latencies = list(
                zip(*[search_one(query) for query in tqdm.tqdm(queries)])
            )
        else:
            ctx = get_context(self.get_mp_start_method())

            with ctx.Pool(
                processes=parallel,
                initializer=self.__class__.init_client,
                initargs=(
                    self.host,
                    distance,
                    self.connection_params,
                    self.search_params,
                ),
            ) as pool:
                if parallel > 10:
                    #time.sleep(15)  # Wait for all processes to start
                    sleep_until_next_target()
                

                
                
                
                
                if oldway:
                    query_batches = list(self.batch_iterable(queries, batch))
                    queries = [item for sublist in query_batches for item in sublist]
                    print("starting")
                    start = time.perf_counter()
                    precisions, latencies = list(
                        zip(*pool.imap_unordered(search_one, iterable=tqdm.tqdm(queries)))
                    )
                else:
                    query_batches = list(self.batch_iterable(queries, batch))
                    print("starting")
                    start = time.perf_counter()
                    precisions, latencies = list(
                        zip(*pool.imap_unordered(search_many, iterable=tqdm.tqdm(query_batches)))
                    )

        total_time = time.perf_counter() - start
        print("ending")
        self.__class__.delete_client()
        if(parallel > 1 and not oldway):
            latencies = [item for sublist in latencies for item in sublist]
        return {
            "total_time": total_time,
            "mean_time": np.mean(latencies),
            "mean_precisions": np.mean(precisions),
            "std_time": np.std(latencies),
            "min_time": np.min(latencies),
            "max_time": np.max(latencies),
            "rps": len(latencies) / total_time,
            "p50_time": np.percentile(latencies, 50),
            "p95_time": np.percentile(latencies, 95),
            "p99_time": np.percentile(latencies, 99),
            "precisions": precisions,
            "latencies": latencies,
        }

    def setup_search(self):
        pass

    def post_search(self):
        pass

    @classmethod
    def delete_client(cls):
        pass
