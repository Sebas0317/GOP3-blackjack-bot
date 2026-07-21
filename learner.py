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

    def _min_samples(self, tc):
        if tc >= 5: return 2000
        if tc >= 3: return 1000
        if tc >= 1: return 500
        return 200

    def best_action(self, hand, dealer, tc, actions):
        sk = self._state_key(hand, dealer, tc)
        best = None
        best_q = float("-inf")
        min_s = self._min_samples(tc)
        for a in actions:
            key = self._action_key(sk, a)
            count = self.counts.get(key, 0)
            if count < min_s:
                continue
            q = self.q.get(key, 0.0)
            if q > best_q:
                best_q = q
                best = a
        return best

    def choose_action(self, hand, dealer, tc, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_action = self.best_action(hand, dealer, tc, actions)
        if q_action is not None:
            return q_action
        return random.choice(actions)

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

    def samples_by_tc_bucket(self):
        buckets = {}
        for key, count in self.counts.items():
            tc = key.split("|")[2] if "|" in key else "?"
            buckets[tc] = buckets.get(tc, 0) + count
        return buckets

    def report_tc_distribution(self):
        buckets = self.samples_by_tc_bucket()
        total = sum(buckets.values())
        print(f"\n=== Sample Distribution by TC Bucket (total: {total}) ===")
        for tc in sorted(buckets.keys(), key=lambda x: (len(x), x)):
            pct = buckets[tc] / total * 100
            print(f"  TC {tc:>3s}: {buckets[tc]:>8d} ({pct:5.1f}%)")
        high_tc = sum(v for k, v in buckets.items() if k in ("5", "6", "7"))
        print(f"\n  TC >= 3 total: {sum(v for k, v in buckets.items() if k in ('3','4','5','6','7')):>8d} ({sum(v for k, v in buckets.items() if k in ('3','4','5','6','7'))/total*100:.1f}%)")
        print(f"  TC >= 5 total: {high_tc:>8d} ({high_tc/total*100:.1f}%)")

    def clear(self):
        self.q = {}
        self.counts = {}
        self.save()
