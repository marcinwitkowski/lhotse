from concurrent.futures import ProcessPoolExecutor


class LhotseDataLoader:
    def __init__(self, dataset, sampler, num_workers: int = 1, prefetch_factor: int = 2) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.pool = ProcessPoolExecutor(num_workers)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._iter = None
        self._futures = []

    def __iter__(self):
        self._iter = iter(self.sampler)
        for _ in range(self.prefetch_factor * self.num_workers):
            self._schedule_one()
        return self

    def _schedule_one(self):
        if self._iter is not None:
            try:
                self._futures.append(
                    self.pool.submit(self.dataset.__getitem__, next(self._iter))
                )
            except StopIteration:
                self._iter = None

    def _retrieve_one(self):
        if self._futures:
            return self._futures.pop().result()
        raise StopIteration()

    def __next__(self):
        self._schedule_one()
        return self._retrieve_one()