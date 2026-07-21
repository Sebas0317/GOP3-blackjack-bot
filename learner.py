import json
import os
import random


class LearningTable:
    def __init__(self, filepath="learning_data.json", learning_rate=0.15, discount=0.95):
        self.filepath = filepath
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = 0.0
        self.q = {}
        self.counts = {}
        self.load()

    def _tc_bucket(self, tc):
        if tc <= -4: return "m4"
        if tc <= -2: return "m2"
        if tc <= -1: return "m1"
        if tc <= 0: return "0"
        if tc <= 1: return "1"
        if tc <= 2: return "2"
        if tc <= 3: return "3"
        if tc <= 4: return "4"
        if tc <= 5: return "5"
        if tc <= 6: return "6"
        return "7"

    def _state_key(self, hand, dealer, tc):
        return f"{hand}|{dealer}|{self._tc_bucket(tc)}"

    def _action_key(self, state_key, action):
        return f"{state_key}|{action}"

    def best_action(self, hand, dealer, tc, actions):
        sk = self._state_key(hand, dealer, tc)
        best = None
        best_q = float("-inf")
        min_samples = 3
        for a in actions:
            key = self._action_key(sk, a)
            count = self.counts.get(key, 0)
            if count < min_samples:
                continue
            q = self.q.get(key, 0.0)
            if q > best_q:
                best_q = q
                best = a
        return best

    def choose_action(self, hand, dealer, tc, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.best_action(hand, dealer, tc, actions)

    def record(self, hand, dealer, tc, action, result):
        reward = 1 if result == "win" else -1 if result == "lose" else 0
        key = self._action_key(self._state_key(hand, dealer, tc), action)
        current = self.q.get(key, 0.0)
        self.q[key] = current + self.lr * (reward - current)
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def total_samples(self):
        return sum(self.counts.values())

    def save(self):
        data = {"q": self.q, "counts": self.counts}
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=1)

    def load(self):
        if not (os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0):
            return
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            if not data:
                return
            if isinstance(data, dict) and "q" in data:
                self.q = {k: float(v) for k, v in data["q"].items()}
                self.counts = {k: int(v) for k, v in data.get("counts", {}).items()}
            elif isinstance(next(iter(data.values())), dict) and "wins" in next(iter(data.values())):
                for k, v in data.items():
                    total = v.get("wins", 0) + v.get("losses", 0) + v.get("draws", 0)
                    if total > 0:
                        wr = v["wins"] / total
                        self.q[k] = 2.0 * wr - 1.0
                        self.counts[k] = total
            else:
                self.q = {k: float(v) for k, v in data.items()}
        except Exception as e:
            print(f"Failed to load {self.filepath}: {e}")

    def clear(self):
        self.q = {}
        self.counts = {}
        self.save()
