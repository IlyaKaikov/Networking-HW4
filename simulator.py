import argparse
import heapq
import random
from collections import deque
from typing import Deque, List, Tuple

ARRIVAL = 0
DEPARTURE = 1

class Event(tuple): # an event tuple is either (arrival_time, ARRIVAL, None) or (departure_time, DEPARTURE, server_index)
    time: float
    kind: int
    srv_idx: int | None

def exp_time(rate: float) -> float:
    return random.expovariate(rate)

class Server:
    def __init__(self, Ni_queue: int, mu: float):
        if Ni_queue < 0:
            raise ValueError("queue length must be >=0")
        if mu <= 0:
            raise ValueError("mu must be positive")
        self.queue: Deque[float] = deque()
        self.busy = False
        self.capacity = Ni_queue + 1
        self.mu = mu
        self.in_system = 0

def run_sim(T: float, probs: List[float], lam: float, queues: List[int], mus: List[float], *, seed: int | None = None) -> Tuple[int, int, float, float, float]:
    if T <= 0:
        raise ValueError("T must be positive")
    M = len(probs)
    if not (len(queues) == len(mus) == M):
        raise ValueError("need M queue sizes and M mus")
    if any(p < 0 for p in probs):
        raise ValueError("probabilities must be >=0")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")
    if lam <= 0:
        raise ValueError("lambda must be positive")

    random.seed(seed)
    servers: List[Server] = [Server(Ni, mu) for Ni, mu in zip(queues, mus)]
    served = 0
    dropped = 0
    total_wait = 0.0
    total_service = 0.0
    events: List[Event] = []
    first = exp_time(lam)
    heapq.heappush(events, (first, ARRIVAL, None))
    now = 0.0

    while events:
        time, kind, srv_idx = heapq.heappop(events)
        now = time
        if kind == ARRIVAL:
            next_time = now + exp_time(lam)
            if next_time <= T: # arrivals stop after T time
                heapq.heappush(events, (next_time, ARRIVAL, None))

            r = random.random() # random range is [0,1)
            cumulative = 0.0
            chosen = 0

            # the following for loop chooses the next server based on the server probabilities and random r value
            # for example: [0, p1=0.25, p1+p2=0.5, r=0.6, p1+p2+p3=0.75, p1+p2+p3+p4=1]
            # server p3 will be chosen because r < cumulative = p1+p2+p3
            for i, p in enumerate(probs):
                cumulative += p
                if r < cumulative:
                    chosen = i
                    break

            srv = servers[chosen]
            if srv.in_system < srv.capacity:
                srv.in_system += 1
                if not srv.busy:
                    srv.busy = True
                    service = exp_time(srv.mu)
                    total_service += service
                    dep_time = now + service
                    heapq.heappush(events, (dep_time, DEPARTURE, chosen))
                else:
                    srv.queue.append(now)
            else:
                dropped += 1

        else:
            srv = servers[srv_idx]
            served += 1
            srv.in_system -= 1
            if srv.queue:
                nxt_arrival = srv.queue.popleft()
                wait = now - nxt_arrival
                total_wait += wait
                service = exp_time(srv.mu)
                total_service += service
                dep_time = now + service
                heapq.heappush(events, (dep_time, DEPARTURE, srv_idx))
            else:
                srv.busy = False

    avg_wait = (total_wait / served) if served else 0.0
    avg_service = (total_service / served) if served else 0.0
    return served, dropped, now, avg_wait, avg_service

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("T", type=float)
    parser.add_argument("M", type=int)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    T = args.T
    M = args.M
    rest = args.rest
    if len(rest) < M + 1 + M + M:
        parser.error("Not enough parameters")

    try:
        probs = [float(x) for x in rest[:M]]
        lam = float(rest[M])
        queues = [int(x) for x in rest[M + 1 : M + 1 + M]]
        mus = [float(x) for x in rest[M + 1 + M : M + 1 + M + M]]
        remaining = rest[M + 1 + M + M :]
        seed = int(remaining[0]) if remaining else None
    except ValueError as e:
        parser.error(f"Invalid numeric value: {e}")

    served, dropped, end_time, avg_wait, avg_service = run_sim(T, probs, lam, queues, mus, seed=seed)
    print(f"{served} {dropped} {end_time:.4f} {avg_wait:.4f} {avg_service:.4f}")

if __name__ == "__main__":
    main()
