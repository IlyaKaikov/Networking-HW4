import heapq
import random
import argparse
from collections import deque
from typing import Tuple, Deque

ARRIVAL = 0
DEPARTURE = 1

def exp_time(rate: float) -> float:
    return random.expovariate(rate)

def run_mm1n(lambda_rate: float, mu_rate: float, capacity: int, horizon: float, *, seed: int | None = None) -> Tuple[int, int, float, float]:
    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("lambda and mu must be positive")
    if capacity < 1:
        raise ValueError("N must be >=1")
    if horizon <= 0:
        raise ValueError("time must be positive")

    if seed is None:
        random.seed()
    else:
        random.seed(seed)

    now = 0.0
    server_busy = False
    in_system = 0
    queue: Deque[float] = deque()
    served = 0
    dropped = 0
    total_wait = 0.0
    events: list[tuple[float, int, float | None]] = []
    first_arrival = exp_time(lambda_rate)
    heapq.heappush(events, (first_arrival, ARRIVAL, None))

    while events:
        time, kind, payload = heapq.heappop(events)
        if time > horizon:
            now = horizon
            break

        now = time
        if kind == ARRIVAL:
            next_arrival = now + exp_time(lambda_rate)
            heapq.heappush(events, (next_arrival, ARRIVAL, None))

            if in_system < capacity:
                in_system += 1
                if not server_busy:
                    server_busy = True
                    service_time = exp_time(mu_rate)
                    departure_time = now + service_time
                    heapq.heappush(events, (departure_time, DEPARTURE, now))
                else:
                    queue.append(now)
            else:
                dropped += 1

        else:
            served += 1
            arrival_time = payload
            total_wait += now - arrival_time
            in_system -= 1
            if queue:
                next_arrival_time = queue.popleft()
                service_time = exp_time(mu_rate)
                departure_time = now + service_time
                heapq.heappush(events, (departure_time, DEPARTURE, next_arrival_time))
            else:
                server_busy = False

    avg_wait = total_wait / served if served else 0.0
    return served, dropped, avg_wait, now

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lam", type=float)
    parser.add_argument("mu", type=float)
    parser.add_argument("capacity", type=int)
    parser.add_argument("time", type=float)
    parser.add_argument("seed", type=int, nargs="?", default=None)
    args = parser.parse_args()
    served, dropped, avg_wait, end_time = run_mm1n(args.lam, args.mu, args.capacity, args.time, seed=args.seed)
    print(f"{served} {dropped} {avg_wait:.4f} {end_time:.4f}")

if __name__ == "__main__":
    main()
