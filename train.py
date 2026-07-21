import random
import sys
from time import time
from learner import LearningTable
from constant import CHEAT_SHEET, HILO_COUNT, I18_DEVIATIONS, DECKS_IN_SHOE

RANKS = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "t", "j", "q", "k"]


def card_value(rank):
    if rank in ("t", "j", "q", "k"):
        return 10
    if rank == "a":
        return 11
    return int(rank)


def hand_value(cards):
    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if c == "a")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def hand_key(cards):
    if len(cards) == 2:
        v1, v2 = card_value(cards[0]), card_value(cards[1])
        if v1 == v2:
            return f"{v1},{v1}" if v1 != 11 else "A,A"
        if "a" in cards:
            other = v2 if cards[0] == "a" else v1
            return f"A,{other}"
        return str(v1 + v2)
    return str(hand_value(cards))


def dealer_str(rank):
    return "A" if rank == "a" else str(card_value(rank))


class Shoe:
    def __init__(self, decks=6, penetration=0.75):
        self.decks = decks
        self.penetration = penetration
        self.cards = []
        self.cards_dealt = 0
        self.shuffle()

    def shuffle(self):
        self.cards = [r for _ in range(self.decks) for _ in range(4) for r in RANKS]
        random.shuffle(self.cards)
        self.cards_dealt = 0

    def draw(self):
        self.cards_dealt += 1
        return self.cards.pop()

    def need_shuffle(self):
        max_dealt = int(len(self.cards) * self.penetration)
        return self.cards_dealt >= max_dealt


class Trainer:
    def __init__(self):
        self.shoe = Shoe(DECKS_IN_SHOE)
        self.running_count = 0
        self.learning = LearningTable()
        self.pending = []
        self.hands_played = 0
        self.results = {"win": 0, "lose": 0, "draw": 0}

    @property
    def decks_remaining(self):
        return max(1.0, DECKS_IN_SHOE - self.shoe.cards_dealt / 52.0)

    @property
    def true_count(self):
        return self.running_count / self.decks_remaining

    def update_count(self, rank):
        self.running_count += HILO_COUNT.get(rank, 0)

    def deal(self):
        rank = self.shoe.draw()
        self.update_count(rank)
        return rank

    def get_strategy(self, h_key, d_str):
        basic = CHEAT_SHEET.get((h_key, d_str), "stand")
        learned = self.learning.best_action(h_key, d_str, self.true_count, ["hit", "stand", "double"])
        if learned is not None:
            return learned
        key = (h_key, d_str)
        if key in I18_DEVIATIONS:
            dev_action, min_tc = I18_DEVIATIONS[key]
            if self.true_count >= min_tc and dev_action in ["hit", "stand", "double"]:
                return dev_action
        return basic

    def record(self, outcome):
        for h, d, tc, a in self.pending:
            self.learning.record(h, d, tc, a, outcome)
        self.pending = []

    def play_hand(self):
        player = [self.deal(), self.deal()]
        dealer_up = self.deal()

        if hand_value(player) == 21:
            dealer_down = self.deal()
            if hand_value([dealer_up, dealer_down]) == 21:
                pass  # push
            else:
                self.results["win"] += 1
            self.hands_played += 1
            return

        dealer_down = self.deal()
        dealer_cards = [dealer_up, dealer_down]

        # Player's turn
        while hand_value(player) < 21:
            hk = hand_key(player)
            ds = dealer_str(dealer_up)
            action = self.get_strategy(hk, ds)
            self.pending.append((hk, ds, self.true_count, action))

            if action == "stand":
                break
            elif action == "double":
                player.append(self.deal())
                break
            else:  # hit
                player.append(self.deal())

        pv = hand_value(player)
        if pv > 21:
            self.record("lose")
            self.results["lose"] += 1
            self.hands_played += 1
            return

        # Dealer's turn
        while hand_value(dealer_cards) < 17:
            dealer_cards.append(self.deal())

        dv = hand_value(dealer_cards)
        if dv > 21 or pv > dv:
            outcome = "win"
        elif pv < dv:
            outcome = "lose"
        else:
            outcome = "draw"

        self.record(outcome)
        self.results[outcome] += 1
        self.hands_played += 1

    def run(self, num_hands, progress=True):
        start = time()
        report_interval = max(1, num_hands // 100)

        for i in range(num_hands):
            if self.shoe.need_shuffle():
                self.shoe.shuffle()
                self.running_count = 0
            self.play_hand()

            if progress and (i + 1) % report_interval == 0:
                pct = (i + 1) / num_hands * 100
                wr = self.results["win"] / max(1, self.hands_played) * 100
                sys.stdout.write(f"\r{pct:.0f}% | hands: {self.hands_played} | win: {wr:.1f}% | samples: {self.learning.total_samples}")
                sys.stdout.flush()

        elapsed = time() - start
        self.learning.save()

        wr = self.results["win"] / max(1, self.hands_played) * 100
        print(f"\nDone in {elapsed:.1f}s | {self.hands_played} hands | win rate: {wr:.1f}% | ML samples: {self.learning.total_samples}")
        print(f"Wins: {self.results['win']} | Losses: {self.results['lose']} | Draws: {self.results['draw']}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    t = Trainer()
    t.run(n)
