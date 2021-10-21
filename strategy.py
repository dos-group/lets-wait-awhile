import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import timedelta
from typing import List, Union

import numpy as np
import pandas as pd
import simpy
from leaf.infrastructure import Node

logger = logging.getLogger(__name__)


class Strategy(ABC):
    replicated_bits = 0

    def __init__(self, node: Node):
        self.node = node

    @abstractmethod
    def run(self, env: simpy.Environment, duration: int):
        pass


class BidirectionalStrategy(Strategy):
    """A strategy for scheduled workloads in the future, i.e. workloads that can be shifted in both "directions"."""

    def __init__(self, node: Node, ci: List[float], window_in_minutes: float, interval: int):
        super().__init__(node)
        self.ci = ci
        self.window_in_minutes = window_in_minutes
        self.interval = interval
        self.forecast_steps = int(window_in_minutes / self.interval)
        self.hist = np.zeros(48)  # 48 steps per day

    def run(self, env: simpy.Environment, duration: int):
        assert duration == self.interval
        timeline_idx = env.now // self.interval
        forecast = self.ci[timeline_idx:timeline_idx + self.forecast_steps * 2 + 1]

        wait_until = np.argmin(forecast)
        timestamp = forecast.index[wait_until]
        minute = 0 if timestamp.minute == 0 else 1
        self.hist[timestamp.hour * 2 + minute] += 1

        logger.debug(f"{env.now}: Wait {wait_until * self.interval}h (current: {self.ci[timeline_idx]}, "
                     f"then {self.ci[timeline_idx + wait_until]}).")
        yield env.timeout(wait_until * self.interval)
        logger.debug(f"{env.now}: Start run")
        self.node.used_mips += 1
        yield env.timeout(duration)
        self.node.used_mips -= 1
        logger.debug(f"{env.now}: Finished run.")


class AdHocStrategy(Strategy):
    """A strategy for ad hoc workloads, i.e. workloads that can be shifted into the future (postponed)."""

    def __init__(self, node: Node, ci: pd.DataFrame, interval: int, forecast: Union[str, int] = None, interruptible: bool = False):
        super().__init__(node)
        self.ci = ci
        self.interval = interval
        self.forecast = forecast
        self.interruptible = interruptible
        self.forecast_steps_hist = defaultdict(int)

    def run(self, env: simpy.Environment, duration: int):
        timeline_idx = env.now // self.interval
        duration_steps = int(duration/self.interval)
        forecast_steps = self._forecast_steps(timeline_idx, duration_steps)

        forecast = self.ci[timeline_idx:timeline_idx + duration_steps + forecast_steps + 1]
        if len(forecast) < duration_steps:
            forecast = self.ci[timeline_idx:timeline_idx + duration_steps]

        if self.interruptible:
            # Get the <duration_steps> points in time with the lowest CI
            worst_ci_to_include = forecast.sort_values()[duration_steps - 1]
            # Get their indices in the forecast dataframe
            lowest_indices = np.where(forecast <= worst_ci_to_include)[0]
            last_index = None

            # Reserve one self.interval of time for every index
            for index in lowest_indices:
                if last_index is None:
                    wait_until = index  # First identified index
                else:
                    wait_until = index - last_index - 1
                last_index = index

                yield env.timeout(wait_until * self.interval)
                self.node.used_mips += 1
                yield env.timeout(self.interval)
                self.node.used_mips -= 1
        else:
            # Calculate mean CI over the duration for every possible start step
            window_means = forecast.rolling(duration_steps).mean()[duration_steps - 1:]
            # The lowest mean will be the start
            wait_until = np.argmin(window_means)

            # Wait until the identified point in time and reserve the full duration
            yield env.timeout(wait_until * self.interval)
            self.node.used_mips += 1
            yield env.timeout(duration)
            self.node.used_mips -= 1

    def _forecast_steps(self, timeline_idx: int, duration_steps: int) -> int:
        if isinstance(self.forecast, (int, float)):
            forecast_steps = int(self.forecast / self.interval)
            self.forecast_steps_hist[forecast_steps] += 1
            return forecast_steps

        # Original finish timestamp
        oft: pd.Timestamp = self.ci.reset_index().loc[timeline_idx + duration_steps, "Time"]
        weekday = oft.weekday()
        time = oft.hour
        if oft.minute == 30:
            time += 0.5
        if self.forecast == "next_workday":
            if weekday == 5:  # Saturday
                finish_timestamp = oft.replace(hour=9, minute=0, second=0) + timedelta(days=2)
            elif weekday == 6 or time > 17:  # Sunday or after 17:00
                finish_timestamp = oft.replace(hour=9, minute=0, second=0) + timedelta(days=1)
            elif time < 9:  # before 9:00
                finish_timestamp = oft.replace(hour=9, minute=0, second=0)
            else:
                finish_timestamp = oft
        elif self.forecast == "next_monday":
            if weekday == 0 and time <= 9:  # Monday before 9.00
                finish_timestamp = oft.replace(hour=9, minute=0, second=0)
            else:
                finish_timestamp = oft.replace(hour=9, minute=0, second=0) + timedelta(days=7-weekday)
        elif self.forecast == "semi_weekly":
            if (weekday == 0 and time > 9) or weekday == 1 or weekday == 2 or (weekday == 3 and time <= 9): # Shift to Thu
                finish_timestamp = oft.replace(hour=9, minute=0, second=0) + timedelta(days=(7-weekday+3) % 7)
            if (weekday == 3 and time > 9) or weekday == 4 or weekday == 5 or weekday == 6 or (weekday == 0 and time <= 9): # Shift to Mo
                finish_timestamp = oft.replace(hour=9, minute=0, second=0) + timedelta(days=(7-weekday) % 7)
            else:
                ValueError("Unexpected original finish timestamp" + str(oft))
        else:
            raise ValueError("Unknown forecast strategy: " + self.forecast)

        delta = finish_timestamp - oft
        forecast_steps = int(delta.days * 48 + delta.seconds / 60 / self.interval)
        self.forecast_steps_hist[forecast_steps] += 1
        return forecast_steps
