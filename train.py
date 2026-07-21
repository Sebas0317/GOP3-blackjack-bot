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
        if "a" in cards:
            return f"A,{v2 if cards[0] == 'a' else v1}"
        if v1 == v2:
            return f"{v1},{v1}"
        return str(v1 + v2)
    h = hand_value(cards)
    if any(c == "a" for c in cards) and h != hand_value([1 if c == "a" else card_value(c) for c in cards]):
        aces = [c for c in cards if c == "a"]
        others = [c for c in cards if c != "a"]
        if len(aces) == 1 and len(others) == 1:
            return f"A,{card_value(others[0])}"
    return str(h)


def dealer_str(rank):
    return "A" if rank == "a" else str(card_value(rank))


def is_pair(cards):
    return len(cards) == 2 and card_value(cards[0]) == card_value(cards[1])


def final_reward(player_value, dealer_value):
    if player_value > 21:
        return -1
    if dealer_value > 21 or player_value > dealer_value:
        return 1
    if player_value < dealer_value:
        return -1
    return 0


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


def dealer_play(shoe, cards):
    while hand_value(cards) < 17:
        cards.append(shoe.deal())
    return cards


def get_action(hk, ds, tc, learning, possible, explore=False):
    if explore:
        return learning.choose_action(hk, ds, tc, possible)
    basic = CHEAT_SHEET.get((hk, ds), "stand")
    learned = learning.best_action(hk, ds, tc, possible)
    if learned is not None:
        return learned
    key = (hk, ds)
    if key in I18_DEVIATIONS:
        dev_action, min_tc = I18_DEVIATIONS[key]
        if tc >= min_tc and dev_action in possible:
            return dev_action
    return basic


def state_action_key(learning, hk, ds, tc, action):
    return f"{hk}|{ds}|{learning._tc_bucket(tc)}|{action}"


def valid_actions(hk, hand_size):
    acts = ["hit", "stand"]
    if hand_size == 2:
        acts.append("double")
    if "," in hk and hand_size == 2:
        acts.append("split")
    return acts


def report_deviations(learning, min_samples=20, top_n=30):
    print("\n=== Learned Deviations vs Basic Strategy ===")
    found = []
    tcs = [-3, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    for tc in tcs:
        for (hk, ds), basic_action in CHEAT_SHEET.items():
            basic_key = state_action_key(learning, hk, ds, tc, basic_action)
            c_basic = learning.counts.get(basic_key, 0)
            if c_basic < min_samples:
                continue
            q_basic = learning.q.get(basic_key, 0.0)
            for alt in valid_actions(hk, 2):
                if alt == basic_action:
                    continue
                alt_key = state_action_key(learning, hk, ds, tc, alt)
                c_alt = learning.counts.get(alt_key, 0)
                if c_alt < min_samples:
                    continue
                q_alt = learning.q.get(alt_key, 0.0)
                if q_alt > q_basic + 0.03:
                    tc_label = learning._tc_bucket(tc)
                    found.append((q_alt - q_basic, f"  TC {tc_label}: ({hk}, {ds}) {basic_action} -> {alt} (d={q_alt - q_basic:+.3f}, b={c_basic}, a={c_alt})"))
    found.sort(reverse=True, key=lambda x: x[0])
    for _, line in found[:top_n]:
        print(line)
    if not found:
        print("  No deviations found (need more exploration)")
    print(f"  Total candidates: {len(found)}")


class Trainer:
    def __init__(self, explore=False, epsilon=0.15):
        self.shoe = Shoe(DECKS_IN_SHOE)
        self.running_count = 0
        self.learning = LearningTable()
        self.learning.epsilon = epsilon
        self.explore = explore
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

    def get_state(self, cards, dealer_up):
        return hand_key(cards), dealer_str(dealer_up)

    def play_single_hand(self, player_cards, dealer_up):
        decisions = []
        while hand_value(player_cards) < 21:
            hk, ds = self.get_state(player_cards, dealer_up)
            tc = self.true_count
            possible = ["hit", "stand"]
            if len(player_cards) == 2:
                possible.append("double")
            action = get_action(hk, ds, tc, self.learning, possible, explore=self.explore)
            if action == "stand":
                decisions.append((hk, ds, tc, "stand"))
                break
            elif action == "double":
                player_cards.append(self.deal())
                decisions.append((hk, ds, tc, "double"))
                break
            else:
                player_cards.append(self.deal())
                decisions.append((hk, ds, tc, "hit"))
                if hand_value(player_cards) > 21:
                    break
        return player_cards, decisions

    def mc_update(self, decisions, R):
        for hk, ds, tc, action in decisions:
            self.learning.record(hk, ds, tc, action, "win" if R == 1 else "lose" if R == -1 else "draw")

    def play_hand(self):
        player = [self.deal(), self.deal()]
        dealer_up = self.deal()

        if hand_value(player) == 21:
            dealer_down = self.deal()
            if hand_value([dealer_up, dealer_down]) != 21:
                self.results["win"] += 1
            self.hands_played += 1
            return

        dealer_down = self.deal()
        dealer_cards = [dealer_up, dealer_down]

        if is_pair(player):
            hk, ds = self.get_state(player, dealer_up)
            tc = self.true_count
            action = get_action(hk, ds, tc, self.learning, ["hit", "stand", "double", "split"], explore=self.explore)
            if action == "split":
                hand1, dec1 = self.play_single_hand([player[0], self.deal()], dealer_up)
                hand2, dec2 = self.play_single_hand([player[1], self.deal()], dealer_up)
                dealer_play(self, dealer_cards)
                dv = hand_value(dealer_cards)
                rewards = []
                for hand, decisions in [(hand1, dec1), (hand2, dec2)]:
                    pv = hand_value(hand)
                    R = final_reward(pv, dv)
                    self.mc_update(decisions, R)
                    rewards.append(R)
                    outcome = "win" if R == 1 else "lose" if R == -1 else "draw"
                    self.results[outcome] += 1
                    self.hands_played += 1
                net = sum(rewards)
                split_outcome = "win" if net > 0 else "lose" if net < 0 else "draw"
                self.learning.record(hk, ds, tc, "split", split_outcome)
                return

        player, decisions = self.play_single_hand(player, dealer_up)
        pv = hand_value(player)
        if pv > 21:
            R = -1
        else:
            dealer_play(self, dealer_cards)
            dv = hand_value(dealer_cards)
            R = final_reward(pv, dv)

        self.mc_update(decisions, R)
        self.hands_played += 1
        outcome = "win" if R == 1 else "lose" if R == -1 else "draw"
        self.results[outcome] += 1

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
    n = 10000
    explore = False
    analyze = False
    for arg in sys.argv[1:]:
        if arg == "--explore":
            explore = True
        elif arg == "--analyze":
            analyze = True
        else:
            try:
                n = int(arg)
            except ValueError:
                pass
    t = Trainer(explore=explore, epsilon=0.15 if explore else 0.0)
    t.run(n)
    if analyze:
        report_deviations(t.learning, min_samples=20)
