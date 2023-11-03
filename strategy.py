import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Union

import numpy as np
import pandas as pd
import simpy
from leaf.infrastructure import Node

from vessim_simulator import TimeSeriesApi

logger = logging.getLogger(__name__)


class Strategy(ABC):
    replicated_bits = 0

    def __init__(self, node: Node, ci_df: pd.Series, interval: int) -> None:
        self.node = node
        self.api = TimeSeriesApi(actual=ci_df)
        self.interval = interval

    @abstractmethod
    def run(self, env: simpy.Environment, duration: int):
        pass

    def _start_time(self, env: simpy.Environment) -> pd.Timestamp:
        return (
            pd.Timestamp(self.api._actual.index[0]) + 
            pd.Timedelta(minutes=env.now)
        )

    def _forecast(self, start_time: datetime, end_time: datetime) -> pd.Series:
        return pd.concat([
            pd.Series(self.api.actual(start_time), index=[start_time]),
            self.api.forecast(start_time, end_time, frequency=f"{self.interval}min")
        ])

class BidirectionalStrategy(Strategy):
    """A strategy for scheduled workloads in the future.
    
    Workloads that can be shifted in both "directions".
    """

    def __init__(
        self, 
        node: Node, 
        ci_df: pd.DataFrame, 
        window_in_minutes: float,
        interval: int
    ):
        super().__init__(node, ci_df, interval)
        self.window_in_minutes = window_in_minutes
        self.hist = np.zeros(48)  # 48 steps per day

    def run(self, env: simpy.Environment, duration: int):
        assert duration == self.interval
        start_time = self._start_time(env)
        end_time = start_time + pd.Timedelta(minutes=self.window_in_minutes * 2)
        forecast = self._forecast(start_time, end_time)

        wait_index = np.argmin(forecast)
        timestamp = forecast.index[wait_index]
        minute = 0 if timestamp.minute == 0 else 1
        self.hist[timestamp.hour * 2 + minute] += 1

        wait_minutes = (timestamp - start_time).total_seconds() / 60
        logger.debug(
            f"{env.now}: Wait {wait_minutes}h (current: {start_time}, "
            f"then {start_time + pd.Timedelta(minutes=wait_minutes)})."
        )
        yield env.timeout(wait_minutes)
        logger.debug(f"{env.now}: Start run")
        self.node.used_cu += 1
        yield env.timeout(duration)
        self.node.used_cu -= 1
        logger.debug(f"{env.now}: Finished run.")


class AdHocStrategy(Strategy):
    """A strategy for ad hoc workloads.
    
    Workloads that can be shifted into the future (postponed).
    """

    def __init__(
        self, 
        node: Node, 
        ci_df: pd.DataFrame, 
        interval: int, 
        forecast_method: Union[str, int] = None, 
        interruptible: bool = False
    ):
        super().__init__(node, ci_df, interval)
        self.forecast_method = forecast_method
        self.interruptible = interruptible
        self.forecast_steps_hist = defaultdict(int)

    def run(self, env: simpy.Environment, duration: int):
        duration_steps = int(duration/self.interval)
        start_time = self._start_time(env)
        forecast_duration = self._forecast_duration(start_time, duration)
        end_time = start_time + pd.Timedelta(minutes=duration + forecast_duration)
        forecast = self._forecast(start_time, end_time)
        if len(forecast) < duration_steps:
            end_time = start_time + pd.Timedelta(minutes=duration)
            forecast = api.forecast(start_time, end_time)

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
                self.node.used_cu += 1
                yield env.timeout(self.interval)
                self.node.used_cu -= 1
        else:
            # Calculate mean CI over the duration for every possible start step
            window_means = forecast.rolling(duration_steps).mean()[duration_steps - 1:]
            # The lowest mean will be the start
            wait_until = np.argmin(window_means)

            # Wait until the identified point in time and reserve the full duration
            yield env.timeout(wait_until * self.interval)
            self.node.used_cu += 1
            yield env.timeout(duration)
            self.node.used_cu -= 1

    def _forecast_duration(self, start_time: pd.Timestamp, duration: int) -> int:
        if isinstance(self.forecast_method, (int, float)):
            self.forecast_steps_hist[int(self.forecast_method / self.interval)] += 1
            return self.forecast_method

        # Original finish timestamp
        oft: pd.Timestamp = start_time + pd.Timedelta(minutes=duration)

        weekday = oft.weekday()
        time = oft.hour
        if oft.minute == 30:
            time += 0.5
        if self.forecast_method == "next_workday":
            if weekday == 5:  # Saturday
                finish_timestamp = (
                    oft.replace(hour=9, minute=0, second=0) +
                    timedelta(days=2)
                )
            elif weekday == 6 or time > 17:  # Sunday or after 17:00
                finish_timestamp = (
                    oft.replace(hour=9, minute=0, second=0) + 
                    timedelta(days=1)
                )
            elif time < 9:  # before 9:00
                finish_timestamp = oft.replace(hour=9, minute=0, second=0)
            else:
                finish_timestamp = oft
        elif self.forecast_method == "next_monday":
            if weekday == 0 and time <= 9:  # Monday before 9.00
                finish_timestamp = oft.replace(hour=9, minute=0, second=0)
            else:
                finish_timestamp = (
                    oft.replace(hour=9, minute=0, second=0) + 
                    timedelta(days=7-weekday)
                )
        elif self.forecast_method == "semi_weekly":
            if (
                (weekday == 0 and time > 9) or 
                weekday == 1 or 
                weekday == 2 or 
                (weekday == 3 and time <= 9)
            ): # Shift to Thu
                finish_timestamp = (
                    oft.replace(hour=9, minute=0, second=0) + 
                    timedelta(days=(7-weekday+3) % 7)
                )
            if (
                (weekday == 3 and time > 9) or 
                weekday == 4 or 
                weekday == 5 or 
                weekday == 6 or 
                (weekday == 0 and time <= 9)
            ): # Shift to Mo
                finish_timestamp = (
                    oft.replace(hour=9, minute=0, second=0) + 
                    timedelta(days=(7-weekday) % 7)
                )
            else:
                ValueError("Unexpected original finish timestamp" + str(oft))
        else:
            raise ValueError("Unknown forecast strategy: " + self.forecast_method)

        delta = finish_timestamp - oft
        forecast_duration = delta.total_seconds() / 60
        self.forecast_steps_hist[int(forecast_duration / self.interval)] += 1
        return forecast_duration
