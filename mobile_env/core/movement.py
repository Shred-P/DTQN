from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np

from mobile_env.core.entities import UserEquipment


class Movement:
    def __init__(
            self, width: float, height: float, seed: int, reset_rng_episode: str, **kwargs
    ):
        self.width, self.height = width, height
        self.reset_rng_episode = reset_rng_episode

        # RNG for movement and initial positions of UEs
        self.seed = seed
        self.rng = None

    def reset(self) -> None:
        """Reset state of movement object after episode ends."""
        # case: movement patterns remain unchanged between episodes
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE at each time step."""
        pass

    @abstractmethod
    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Reset position of UE e.g. after episode ends."""
        pass


class RandomWaypointMovement(Movement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # track waypoints and initial positions per UE
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = None
        self.initial: Dict[UserEquipment, Tuple[float, float]] = None

    def reset(self) -> None:
        super().reset()
        # NOTE: if RNG is not resetted after episode ends,
        # initial positions will differ between episodes
        self.waypoints = {}
        self.initial = {}

    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE a step towards the random waypoint."""
        # generate random waypoint if UE has none so far
        if ue not in self.waypoints:
            wx = self.rng.uniform(0, self.width)
            wy = self.rng.uniform(0, self.height)
            self.waypoints[ue] = (wx, wy)

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.waypoints[ue])

        # if already close enough to waypoint, move directly onto waypoint
        if np.linalg.norm(position - waypoint) <= ue.velocity:
            # remove waypoint from dict after it has been reached
            waypoint = self.waypoints.pop(ue)
            return waypoint

        # else move by self.velocity towards waypoint
        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)

        return tuple(position)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Return initial position of UE at the beginning of the episode."""
        if ue not in self.initial:
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            self.initial[ue] = (x, y)
            # print(f"UE{ue.ue_id} init at:({x},{y}")
        x, y = self.initial[ue]
        return x, y

class ArticleRandomWaypointMovement(Movement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track waypoints, initial positions, and pause counters per UE
        self.waypoints: Dict[UserEquipment, Tuple[float, float]] = None
        self.initial: Dict[UserEquipment, Tuple[float, float]] = None
        self.pause_counters: Dict[UserEquipment, int] = {}  # New pause counter

    def reset(self) -> None:
        super().reset()
        self.waypoints = {}
        self.initial = {}
        self.pause_counters = {}  # Reset pause counters at the start

    def move(self, ue: UserEquipment) -> Tuple[float, float]:
        """Move UE a step towards the random waypoint, pausing at the waypoint for 2 time steps."""
        # Set speed as a random value between 1 and 3 m/s
        ue.velocity = self.rng.uniform(1.0, 3.0)

        # Check if UE has reached a waypoint
        if ue not in self.waypoints:
            # Generate a new waypoint if UE has none
            wx = self.rng.uniform(0, self.width)
            wy = self.rng.uniform(0, self.height)
            self.waypoints[ue] = (wx, wy)

        position = np.array([ue.x, ue.y])
        waypoint = np.array(self.waypoints[ue])

        # Check if UE has reached its waypoint
        if np.linalg.norm(position - waypoint) <= ue.velocity:
            # Start pause counter if it hasn't been set
            if ue not in self.pause_counters:
                self.pause_counters[ue] = 0

            # Pause for 2 time steps at the waypoint
            if self.pause_counters[ue] < 2:
                self.pause_counters[ue] += 1
                return tuple(position)  # UE stays at the current position
            else:
                # Reset the pause counter and generate a new waypoint
                self.pause_counters.pop(ue)
                self.waypoints.pop(ue)
                return tuple(waypoint)

        # Move UE towards the waypoint if not in pause state
        v = waypoint - position
        position = position + ue.velocity * v / np.linalg.norm(v)

        return tuple(position)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Return initial position of UE at the beginning of the episode."""
        if ue not in self.initial:
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            self.initial[ue] = (x, y)
        x, y = self.initial[ue]
        return x, y

