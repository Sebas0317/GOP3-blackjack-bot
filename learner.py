import json
import os


class LearningTable:
    def __init__(self, filepath="learning_data.json"):
        self.filepath = filepath
        self.data = {}
        self.load()

    def _key(self, hand, dealer, tc_bucket, action):
        return f"{hand}|{dealer}|{tc_bucket}|{action}"

    def _tc_bucket(self, tc):
        if tc <= -3: return "L3"
        elif tc == -2: return "L2"
        elif tc == -1: return "L1"
        elif tc == 0: return "0"
        elif tc == 1: return "1"
        elif tc == 2: return "2"
        elif tc == 3: return "3"
        else: return "H4"

    def record(self, hand, dealer, tc, action, result):
        bucket = self._tc_bucket(tc)
        key = self._key(hand, dealer, bucket, action)
        entry = self.data.setdefault(key, {"wins": 0, "losses": 0, "draws": 0})
        if result == "win":
            entry["wins"] += 1
        elif result == "draw":
            entry["draws"] += 1
        elif result == "lose":
            entry["losses"] += 1
        if sum(entry.values()) % 50 == 0:
            self.save()

    def win_rate(self, hand, dealer, tc, action, min_samples=10):
        bucket = self._tc_bucket(tc)
        entry = self.data.get(self._key(hand, dealer, bucket, action))
        if not entry:
            entry = self.data.get(self._key(hand, dealer, "any", action))
            if not entry:
                return None
        total = entry["wins"] + entry["losses"] + entry["draws"]
        if total < min_samples:
            return None
        return entry["wins"] / total

    def best_action(self, hand, dealer, tc, actions):
        best = None
        best_rate = -1
        for action in actions:
            rate = self.win_rate(hand, dealer, tc, action)
            if rate is not None and rate > best_rate:
                best_rate = rate
                best = action
        return best

    @property
    def total_samples(self):
        return sum(e["wins"] + e["losses"] + e["draws"] for e in self.data.values())

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=1)

    def load(self):
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            with open(self.filepath, "r") as f:
                self.data = json.load(f)

    def clear(self):
        self.data = {}
        self.save()
