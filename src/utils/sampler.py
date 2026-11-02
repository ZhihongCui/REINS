import json
import yaml
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict, Counter


# modules/sampler.py

class Sampler:
    def __init__(self, env, config_path="config/variable_config.yaml"):
        self.env = self._fully_unwrap(env)
        with open(config_path, "r") as f:
            self.variable_template = yaml.safe_load(f)
        self._episode_weather = None
        self._last_action_id = None

    def _fully_unwrap(self, env):
        while hasattr(env, "unwrapped") and env is not env.unwrapped:
            env = env.unwrapped
        return env

    def _sample_weather(self) -> Dict[str, Any]:
        """
        Sample the weather and its derived parameter ranges (visibility, road friction, time of day) for the current episode.
        The returned ranges are used in _extract_value to ensure consistency within the episode while allowing slight frame-to-frame variations.
        """
        kinds = ["clear", "light_rain", "heavy_rain", "fog", "snow"]
        probs = [0.55, 0.20, 0.10, 0.10, 0.05]
        wt = np.random.choice(kinds, p=probs)

        # Set reasonable visibility/friction ranges based on weather
        if wt == "clear":
            vis_range = (120.0, 220.0)      # m
            fric_range = (0.90, 1.00)
            tod_probs = {"day": 0.7, "night": 0.3}
        elif wt == "light_rain":
            vis_range = (70.0, 120.0)
            fric_range = (0.75, 0.90)
            tod_probs = {"day": 0.6, "night": 0.4}
        elif wt == "heavy_rain":
            vis_range = (40.0, 80.0)
            fric_range = (0.60, 0.80)
            tod_probs = {"day": 0.55, "night": 0.45}
        elif wt == "fog":
            vis_range = (20.0, 60.0)
            fric_range = (0.85, 0.95)       # Fog has minimal impact on traction
            tod_probs = {"day": 0.5, "night": 0.5}
        else:  # snow
            vis_range = (30.0, 90.0)
            fric_range = (0.30, 0.60)
            tod_probs = {"day": 0.6, "night": 0.4}

        # Fix a time_of_day, consistent throughout the episode
        tod = np.random.choice(["day", "night"], p=[tod_probs["day"], tod_probs["night"]])

        return {
            "weather_type": wt,
            "visibility_range": vis_range,
            "friction_range": fric_range,
            "time_of_day": tod,
        }

    def _sample_from_range(self, lo: float, hi: float, jitter_ratio: float = 0.05) -> float:
        """
        Uniformly sample from [lo, hi], then add a small relative jitter (default +/-5%) for frame-to-frame variation.
        """
        base = np.random.uniform(lo, hi)
        jitter = base * np.random.uniform(-jitter_ratio, jitter_ratio)
        return float(np.clip(base + jitter, min(lo, hi), max(lo, hi)))

    def _neighbors(self) -> Dict[str, tuple]:
        """
        Returns:
        {
            "front": (veh, dist), "rear": (veh, dist),
            "lf": (veh, dist), "lr": (veh, dist),
            "rf": (veh, dist), "rr": (veh, dist)
        }
        dist is the longitudinal distance along the lane direction (meters), None indicates not found
        """
        ego = self.env.vehicle
        road = self.env.road

        def lane_s(local_lane_index):
            try:
                lane = road.lane(local_lane_index)
                s, _ = lane.local_coordinates(ego.position)
                return s
            except Exception:
                return None

        res = {
            "front": (None, None), "rear": (None, None),
            "lf": (None, None), "lr": (None, None),
            "rf": (None, None), "rr": (None, None),
        }

        try:
            # Ego's current lane
            ei = ego.lane_index  # (road, idx, lane_id)
            ego_lane_id = ei[2]
            ego_s = lane_s(ei)

            # Iterate through all vehicles on the road, find nearest front/rear in same lane and adjacent lanes
            for v in road.vehicles:
                if v is ego or not hasattr(v, "lane_index"):
                    continue
                vi = v.lane_index
                lane_id = vi[2]

                # Only consider vehicles in the same road segment
                if vi[0] != ei[0] or vi[1] != ei[1]:
                    continue

                v_s = None
                try:
                    lane = road.lane(vi)
                    v_s, _ = lane.local_coordinates(v.position)
                except Exception:
                    continue

                if ego_s is None or v_s is None:
                    continue

                d = float(v_s - ego_s)  # Positive: ahead; Negative: behind

                def keep_min_front(slot, veh, dist):
                    # Keep the closer front vehicle (dist > 0)
                    cur = res[slot]
                    if cur[1] is None or dist < cur[1]:
                        res[slot] = (veh, dist)

                def keep_min_rear(slot, veh, dist):
                    # Keep the closer rear vehicle (dist < 0, store positive value)
                    cur = res[slot]
                    dpos = abs(dist)
                    if cur[1] is None or dpos < cur[1]:
                        res[slot] = (veh, dpos)

                if lane_id == ego_lane_id:
                    if d > 0:
                        keep_min_front("front", v, d)
                    elif d < 0:
                        keep_min_rear("rear", v, d)

                elif lane_id == ego_lane_id - 1:  # Left lane
                    if d > 0:
                        keep_min_front("lf", v, d)
                    elif d < 0:
                        keep_min_rear("lr", v, d)

                elif lane_id == ego_lane_id + 1:  # Right lane
                    if d > 0:
                        keep_min_front("rf", v, d)
                    elif d < 0:
                        keep_min_rear("rr", v, d)

        except Exception as e:
            # Return default None on error
            pass

        # Convert None distances to more usable units (keep None, let caller handle as inf/1e6)
        return res
    def _cutin_risk_from_neighbors(self, nb, ego) -> float:
        """
        Estimate cut-in risk using front/rear vehicle distances and speeds in adjacent lanes, returns [0,1]
        Smaller gap, faster approaching rear vehicle, slower front vehicle -> higher risk
        """
        def sp(v): return getattr(v, "speed", 0.0) if v is not None else 0.0
        ego_v = float(getattr(ego, "speed", 0.0))

        # Distance (meters, None -> large number), Speed (m/s)
        lf_d = nb["lf"][1] if nb["lf"][1] is not None else 1e6
        lr_d = nb["lr"][1] if nb["lr"][1] is not None else 1e6
        rf_d = nb["rf"][1] if nb["rf"][1] is not None else 1e6
        rr_d = nb["rr"][1] if nb["rr"][1] is not None else 1e6

        lf_v = sp(nb["lf"][0]); lr_v = sp(nb["lr"][0])
        rf_v = sp(nb["rf"][0]); rr_v = sp(nb["rr"][0])

        # Available gap (smaller is more dangerous)
        gap_L = min(lf_d, lr_d)
        gap_R = min(rf_d, rr_d)

        def norm_gap(g):  # Linear increase within 20m
            return float(np.clip((20.0 - g) / 20.0, 0.0, 1.0))

        # Rear vehicle closing (>0 means rear vehicle is faster), normalized to 10 m/s scale
        close_L = max(0.0, lr_v - ego_v) / 10.0
        close_R = max(0.0, rr_v - ego_v) / 10.0

        # Front vehicle slower (>0 means ego is faster), normalized to 10 m/s scale
        slow_front_L = max(0.0, ego_v - lf_v) / 10.0
        slow_front_R = max(0.0, ego_v - rf_v) / 10.0

        w_gap, w_close, w_slow = 0.6, 0.3, 0.3
        rL_raw = w_gap * norm_gap(gap_L) + w_close * np.clip(close_L, 0, 1) + w_slow * np.clip(slow_front_L, 0, 1)
        rR_raw = w_gap * norm_gap(gap_R) + w_close * np.clip(close_R, 0, 1) + w_slow * np.clip(slow_front_R, 0, 1)

        # Compress to 0~1, reduce linear overflow
        rL = 1.0 - float(np.exp(-rL_raw))
        rR = 1.0 - float(np.exp(-rR_raw))
        return float(np.clip(max(rL, rR), 0.0, 1.0))

    # -----------------------------
    # Main function: Multi-Episode Sampling
    # -----------------------------
    def sample_multiple_episodes(self, n: int, base_seed: int = 42) -> List[Dict[str, Any]]:
        results = []
        print(f"[Sampler] Starting sampling of {n} episodes...")
        for ep in range(n):
            print(f"[Sampler] Episode {ep + 1}/{n}", end="\r", flush=True)
            # Set different random seeds
            current_seed = base_seed + ep
            np.random.seed(current_seed)
            import random, torch
            random.seed(current_seed)
            torch.manual_seed(current_seed)

            # Pass seed when resetting
            obs, _ = self.env.reset(seed=current_seed)
            self._episode_weather = self._sample_weather()
            done = False
            episode_traj = []

            while not done:
                # Uniformly sample from five action types
                possible_actions = [0, 1, 2, 3, 4]  # LANE_LEFT, LANE_RIGHT, IDLE, FASTER, SLOWER
                action = np.random.choice(possible_actions)

                # Inject slight perturbation to enhance dynamics
                if np.random.rand() < 0.2:
                    if hasattr(self.env.vehicle, "steering"):
                        self.env.vehicle.steering += np.random.uniform(-0.1, 0.1)
                    if hasattr(self.env.vehicle, "acceleration"):
                        self.env.vehicle.acceleration += np.random.uniform(-1.0, 1.0)

                obs, _, terminated, truncated, _ = self.env.step(action)
                sample = self.sample_episode(ep, obs,last_action_id=action)
                episode_traj.append(sample)
                done = terminated or truncated

            results.append({"trajectory": episode_traj, "seed": current_seed})

        print()  # Newline after progress indicator
        print(f"[Sampler] Sampled {n} episodes with seeds {base_seed}~{base_seed + n - 1}")
        return results

    # -----------------------------
    # Per-frame sampling content
    # -----------------------------
    def sample_episode(self, episode_id: int, obs: Dict[str, Any], last_action_id: int | None = None) -> Dict[str, Any]:
        if self._episode_weather is None:  # Initialize weather on first call during inference
            self._episode_weather = self._sample_weather()
        self._last_action_id = int(last_action_id) if last_action_id is not None else None
        self._cached_neighbors = None  # Clear neighbor cache for this frame to avoid cross-frame misuse

        sample = {macro: {} for macro in self.variable_template}
        for macro, micro_list in self.variable_template.items():
            for micro in micro_list:
                sample[macro][micro] = self._extract_value(obs, micro)
        sample["episode_id"] = episode_id
        return sample


    # -----------------------------
    # Value extraction logic for each variable
    # -----------------------------
    def _extract_value(self, obs: Dict[str, Any], micro_var: str) -> Any:
        try:
            ego = self.env.vehicle
            road = self.env.road

            # =====================================================
            # Neighbor cache: same lane (front/rear) + adjacent lanes (lf/lr/rf/rr)
            # =====================================================
            nb = getattr(self, "_cached_neighbors", None)
            if nb is None:
                nb = self._neighbors()
                self._cached_neighbors = nb

            def speed_of(v):
                return getattr(v, "speed", 0.0) if v is not None else 0.0

            # -------------------- Traffic: Same lane --------------------
            if micro_var == "headway_distance":
                return nb["front"][1] if nb["front"][1] is not None else 1e6
            elif micro_var == "lead_speed":
                return speed_of(nb["front"][0])
            elif micro_var == "lead_rel_speed":
                return speed_of(nb["front"][0]) - ego.speed
            elif micro_var == "lead_ttc":
                d  = nb["front"][1] if nb["front"][1] is not None else 1e6
                dv = ego.speed - speed_of(nb["front"][0])  # Closing speed (>0 means approaching)
                # Use 99.0 to represent "very large TTC", avoiding np.inf
                return d / max(1e-3, dv) if dv > 0 else 99.0
            elif micro_var == "lead_thw":
                d = nb["front"][1] if nb["front"][1] is not None else 1e6
                return d / max(1e-3, ego.speed) if ego.speed > 1e-3 else 99.0

            elif micro_var == "rear_distance":     return nb["rear"][1] if nb["rear"][1] is not None else 1e6
            elif micro_var == "rear_speed":        return speed_of(nb["rear"][0])
            elif micro_var == "rear_rel_speed":    return ego.speed - speed_of(nb["rear"][0])
            elif micro_var == "rear_ttc":
                d  = nb["rear"][1] if nb["rear"][1] is not None else 1e6
                dv = speed_of(nb["rear"][0]) - ego.speed   # Rear vehicle closing speed (>0 is meaningful)
                return d / max(1e-3, dv) if dv > 0 else 99.0

            # -------------------- Traffic: Left/Right adjacent lanes --------------------
            elif micro_var == "left_front_distance":   return nb["lf"][1] if nb["lf"][1] is not None else 1e6
            elif micro_var == "left_front_speed":      return speed_of(nb["lf"][0])
            elif micro_var == "left_rear_distance":    return nb["lr"][1] if nb["lr"][1] is not None else 1e6
            elif micro_var == "left_rear_speed":       return speed_of(nb["lr"][0])

            elif micro_var == "right_front_distance":  return nb["rf"][1] if nb["rf"][1] is not None else 1e6
            elif micro_var == "right_front_speed":     return speed_of(nb["rf"][0])
            elif micro_var == "right_rear_distance":   return nb["rr"][1] if nb["rr"][1] is not None else 1e6
            elif micro_var == "right_rear_speed":      return speed_of(nb["rr"][0])

            # Lane density (vehicle count in left/right lanes)
            elif micro_var == "lane_density_left":
                return float(self._estimate_lane_density(-1))
            elif micro_var == "lane_density_right":
                return float(self._estimate_lane_density(+1))


            # -------------------- Available gap for lane change (larger is easier) --------------------
            elif micro_var == "available_gap_left":
                f = nb["lf"][1] if nb["lf"][1] is not None else 0.0
                r = nb["lr"][1] if nb["lr"][1] is not None else 0.0
                return min(f, r)
            elif micro_var == "available_gap_right":
                f = nb["rf"][1] if nb["rf"][1] is not None else 0.0
                r = nb["rr"][1] if nb["rr"][1] is not None else 0.0
                return min(f, r)
            elif micro_var == "cutin_risk":
                # Heuristic 0~1 risk score based on neighbors/gaps
                nb = getattr(self, "_cached_neighbors", None)
                if nb is None:
                    nb = self._neighbors()
                    self._cached_neighbors = nb
                return self._cutin_risk_from_neighbors(nb, ego)


            # ===== Environment =====
            elif micro_var == "weather_type":
                return (self._episode_weather or {}).get("weather_type", "clear")

            elif micro_var == "visibility":
                w = self._episode_weather or {}
                lo, hi = w.get("visibility_range", (80.0, 120.0))
                return self._sample_from_range(lo, hi)

            elif micro_var == "road_friction":
                w = self._episode_weather or {}
                lo, hi = w.get("friction_range", (0.8, 1.0))
                return self._sample_from_range(lo, hi)

            elif micro_var == "time_of_day":
                w = self._episode_weather or {}
                return w.get("time_of_day", np.random.choice(["day", "night"]))

            elif micro_var == "speed_limit":
                sl = getattr(ego, "speed_limit", None)
                return float(sl() if callable(sl) else (sl if isinstance(sl, (int,float)) else 30.0))


            # ===== Ego (position & velocity components) =====
            elif micro_var == "ego_x":
                # highway-env: ego.position = np.array([x, y])
                pos = getattr(ego, "position", None)
                return float(pos[0]) if pos is not None else 0.0

            elif micro_var == "ego_y":
                pos = getattr(ego, "position", None)
                return float(pos[1]) if pos is not None else 0.0

            elif micro_var == "ego_vx" or micro_var == "ego_vy":
                # Prefer vector velocity; if unavailable, decompose from speed+heading
                vel = getattr(ego, "velocity", None)
                if vel is not None and len(vel) >= 2:
                    vx, vy = float(vel[0]), float(vel[1])
                else:
                    speed = float(getattr(ego, "speed", 0.0))
                    heading = float(getattr(ego, "heading", 0.0))
                    # highway-env heading is in radians, x direction along lane tangent
                    vx, vy = speed * np.cos(heading), speed * np.sin(heading)
                return vx if micro_var == "ego_vx" else vy

            elif micro_var == "ego_speed":
                return getattr(ego, "speed", 0.0)

            elif micro_var == "ego_acceleration":
                val = getattr(ego, "acceleration", 0.0)
                if abs(val) < 1e-3:
                    val += np.random.uniform(-0.1, 0.1)
                return val

            elif micro_var == "yaw_rate":
                if hasattr(ego, "steering"):
                    return float(ego.steering) * 0.5
                return np.random.uniform(-0.05, 0.05)

            elif micro_var == "jerk":
                return np.random.uniform(-0.5, 0.5)

            elif micro_var == "lane_id":
                return ego.lane_index[2] if hasattr(ego, "lane_index") else -1

            elif micro_var == "lateral_offset":
                lane = road.lane(ego.lane_index) if hasattr(road, "lane") else None
                return lane.local_coordinates(ego.position)[1] if lane else 0.0

            # ===== Action =====
            elif micro_var == "action_type":
                # Prefer using action id passed from run_episode
                aid = getattr(self, "_last_action_id", None)

                # id -> name mapping (Highway-env MetaAction)
                action_map = ["LANE_LEFT", "LANE_RIGHT", "IDLE", "FASTER", "SLOWER"]

                if isinstance(aid, int) and 0 <= aid < len(action_map):
                    return action_map[aid]

                # Fallback: if not passed, try to get from env; if unavailable, return IDLE
                if hasattr(self.env, "action") and isinstance(self.env.action, int):
                    aid2 = self.env.action
                    if 0 <= aid2 < len(action_map):
                        return action_map[aid2]
                return "IDLE"

            # ===== Outcome =====
            elif micro_var == "risk_score":
                speed = getattr(ego, "speed", 0.0)
                yaw = getattr(ego, "steering", 0.0)
                headway = self._extract_value(obs, "headway_distance")
                risk = np.clip((abs(yaw) * 0.5 + max(0, 30 - headway) / 30 + speed / 40), 0, 1)
                return risk

            elif micro_var == "comfort_score":
                jerk = np.random.uniform(-0.5, 0.5)
                acc = getattr(ego, "acceleration", 0.0)
                return np.clip(1.0 - abs(acc) - abs(jerk), 0.0, 1.0)

            elif micro_var == "energy_consumption":
                speed = ego.speed
                acc = getattr(ego, "acceleration", 0.5)
                return np.clip(abs(speed * acc) / 100.0, 0.0, 2.0)

            elif micro_var == "success_flag":
                return int(self.env.done and not getattr(ego, "crashed", False))

            elif micro_var == "violation_flag":
                return int(getattr(ego, "crashed", False))

            else:
                return None

        except Exception as e:
            print(f"[Sampler] Failed to extract {micro_var}: {e}")
            return None

    # -----------------------------
    # Helper functions
    # -----------------------------
    def _get_lead_vehicle(self):
        try:
            ego = self.env.vehicle
            return ego.road.closest_vehicles(ego).get("front")
        except Exception:
            return None

    def _estimate_lane_density(self, direction: int) -> float:
        ego = self.env.vehicle
        try:
            ego_lane = ego.lane_index[2]
            target_lane = ego_lane + direction
            vehicles = [
                v for v in self.env.road.vehicles
                if hasattr(v, "lane_index") and v.lane_index[2] == target_lane
            ]
            return len(vehicles)
        except:
            return 0

    # -----------------------------
    # Save sampling results
    # -----------------------------
    def save_samples(self, samples: List[Dict[str, Any]], path: str):
        with open(path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"[Sampler] Saved {len(samples)} episodes to {path}")
