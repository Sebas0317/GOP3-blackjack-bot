from time import sleep
from numpy import where as np_where
from numpy import sqrt as np_sqrt
import pyautogui
from cv2 import resize, matchTemplate, TM_CCOEFF_NORMED, minMaxLoc
from PyQt5.QtCore import QThread, pyqtSignal
from utils import safe_imread, grab_screen
from constant import NUMBER, COLOR, CHEAT_SHEET, compute_layout, BET_AMOUNT, HILO_COUNT, I18_DEVIATIONS, INSURANCE_TC, DECKS_IN_SHOE
from learner import LearningTable


def is_close(pt1, pt2, threshold=8):
    return np_sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) < threshold


def card_num_from_card_name(card_name):
    if card_name[1] in ["t", "j", "q", "k"]:
        return 10
    elif card_name[1] == "a":
        return 11
    else:
        return int(card_name[1])


def card_num_str_from_card_name(card_name):
    if card_name[1] in ["t", "j", "q", "k"]:
        return "10"
    elif card_name[1] == "a":
        return "A"
    else:
        return card_name[1]


def card_suite_from_two_card_num(card_num1: int, card_num2: int) -> str:
    if card_num1 == card_num2:
        if card_num1 == 11 and card_num2 == 11:
            return "A,A"
        return str(card_num1) + "," + str(card_num2)
    elif card_num1 == 11 or card_num2 == 11:
        if card_num1 == 11 and card_num2 == 11:
            return "A,A"
        elif card_num1 == 11:
            return "A," + str(card_num2)
        else:
            return "A," + str(card_num1)
    else:
        return str(card_num1 + card_num2)


class ProgramThread(QThread):
    statUpdated = pyqtSignal(float, str)
    roundInformUpdated = pyqtSignal(str, str, str)
    countUpdated = pyqtSignal(int, float, float)
    learnUpdated = pyqtSignal(int)

    def __init__(self, bet_amount, language, resolution="1920x1080",
                 martingale=False, martingale_max_steps=4,
                 card_counting=False,
                 wonging=False, wong_tc_threshold=-2,
                 learning=False,
                 kelly_betting=False, bankroll=1000):
        super().__init__()
        self.bet_amount = bet_amount
        self.language = language
        self.running = True
        width, height = (int(x) for x in resolution.split("x"))
        self.layout = compute_layout(width, height)
        self.card_images = {}
        self.image_prefix = "image/" + self.language + "/"
        self.martingale = martingale
        self.martingale_max_steps = martingale_max_steps
        self.card_counting = card_counting
        self.wonging = wonging
        self.wong_tc_threshold = wong_tc_threshold
        self.learning = learning
        self.kelly_betting = kelly_betting
        self.bankroll = bankroll
        self.learning_table = LearningTable()
        self._pending_decisions = []
        self._pending_round = 0
        self._pending_first_outcome = None
        self.running_count = 0
        self.cards_seen = 0
        self.martingale_step = 0
        self.is_doubled = False
        self.split_round = 0
        self.split_active = False
        self.bet_keys = list(BET_AMOUNT.keys())
        for num in NUMBER:
            for col in COLOR:
                card_name = col + num
                card_image = safe_imread(r"image/card/" + card_name + ".png", 0)
                self.card_images[card_name] = card_image

    def compare(self, target, screen):
        res = matchTemplate(screen, target, TM_CCOEFF_NORMED)
        _, val, _, loc = minMaxLoc(res)
        return loc if val >= 0.9 else None

    def clickz(self, top_left):
        x = top_left[0] + self.layout["BUTTON_WIDTH"] / 2
        y = top_left[1] + self.layout["BUTTON_HEIGHT"] / 2
        pyautogui.click(x, y, button="left", duration=0.05)
        pyautogui.moveTo(self.layout["WINDOW_WIDTH"] / 2, self.layout["WINDOW_HEIGHT"] / 2, duration=0.05)

    def update_count(self, card_name):
        if not self.card_counting:
            return
        self.cards_seen += 1
        rank = card_name[1]
        if rank in HILO_COUNT:
            self.running_count += HILO_COUNT[rank]

    def reset_count(self):
        self.running_count = 0
        self.cards_seen = 0
        self.emit_count()

    def emit_count(self):
        if self.card_counting:
            self.countUpdated.emit(self.running_count, round(self.true_count, 1), round(self.decks_remaining, 1))

    @property
    def decks_remaining(self):
        return max(1.0, DECKS_IN_SHOE - self.cards_seen / 52.0)

    @property
    def true_count(self):
        return self.running_count / self.decks_remaining

    def bet_multiplier(self):
        if not self.card_counting:
            return 1
        tc = self.true_count
        if tc >= 6: return 16
        elif tc >= 5: return 10
        elif tc >= 4: return 6
        elif tc >= 3: return 3
        elif tc >= 2: return 2
        elif tc >= 1: return 1.5
        return 1

    def kelly_bet_multiplier(self):
        if not self.card_counting:
            return 1
        tc = self.true_count
        edge = -0.005 + tc * 0.005
        if edge <= 0:
            return 1
        variance = 1.21
        kelly_frac = 0.5 * edge / variance
        return max(1, kelly_frac * self.bankroll)

    def get_adjusted_strategy(self, hand_key, dealer_card_num_str, basic_strategy, possible_actions=None):
        if not self.card_counting:
            return basic_strategy
        key = (hand_key, dealer_card_num_str)
        if key in I18_DEVIATIONS:
            dev_action, min_tc = I18_DEVIATIONS[key]
            if self.true_count >= min_tc:
                basic_strategy = dev_action
        if self.learning and possible_actions:
            learned = self.learning_table.best_action(hand_key, dealer_card_num_str, self.true_count, possible_actions)
            if learned is not None:
                return learned
        return basic_strategy

    def current_bet_key(self):
        base_val = BET_AMOUNT[self.bet_amount]
        total_mult = 1.0
        if self.martingale and self.martingale_step > 0:
            total_mult *= (2 ** self.martingale_step)
        if self.card_counting:
            if self.kelly_betting:
                total_mult *= self.kelly_bet_multiplier()
            else:
                total_mult *= self.bet_multiplier()
        target = base_val * total_mult
        best = self.bet_amount
        best_diff = float("inf")
        for k in self.bet_keys:
            diff = abs(BET_AMOUNT[k] - target)
            if diff < best_diff:
                best_diff = diff
                best = k
        return best

    def _handle_result(self, amount_rate, outcome):
        self.statUpdated.emit(amount_rate, outcome)
        if self.learning and self._pending_decisions:
            outcome_key = "win" if outcome == "win" else "lose" if outcome == "lose" else "draw"
            if self.split_active and self.split_round == 1:
                self._pending_first_outcome = outcome_key
                self._pending_round = 2
            elif self.split_active and self.split_round == 2:
                remaining = []
                for entry in self._pending_decisions:
                    h, d, tc, a, rnd = entry
                    actual = self._pending_first_outcome if rnd <= 1 else outcome_key
                    self.learning_table.record(h, d, tc, a, actual)
                self._pending_decisions.clear()
                self._pending_first_outcome = None
                self._pending_round = 0
                self.learning_table.save()
                self.learnUpdated.emit(self.learning_table.total_samples)
            else:
                while self._pending_decisions:
                    h, d, tc, a, rnd = self._pending_decisions.pop(0)
                    self.learning_table.record(h, d, tc, a, outcome_key)
                self.learning_table.save()
                self.learnUpdated.emit(self.learning_table.total_samples)
        if self.split_active and self.split_round == 2:
            self.split_active = False
            self.split_round = 0
            self.is_doubled = False
        elif self.split_active:
            self.split_round = 2
        else:
            self.is_doubled = False
            if outcome == "win":
                if self.martingale:
                    self.martingale_step = 0
            elif outcome == "lose":
                if self.martingale:
                    self.martingale_step = min(self.martingale_step + 1, self.martingale_max_steps)
        self.emit_count()

    def run(self):
        self.is_doubled = False
        self.split_round = 0
        self.split_active = False

        win = safe_imread(self.image_prefix + "win.png", 0)
        lose = safe_imread(self.image_prefix + "lose.png", 0)
        bust = safe_imread(self.image_prefix + "bust.png", 0)
        draw = safe_imread(self.image_prefix + "draw.png", 0)
        double = safe_imread(self.image_prefix + "double.png", 0)
        stand = safe_imread(self.image_prefix + "stand.png", 0)
        blackjack = safe_imread(self.image_prefix + "blackjack.png", 0)
        bet = safe_imread(r"image/bet/bet" + self.current_bet_key() + ".png", 0)

        while self.running:
            screen = grab_screen()
            screen = resize(screen, (self.layout["WINDOW_WIDTH"], self.layout["WINDOW_HEIGHT"]))

            if self.compare(win, screen):
                amount_rate = 2 if self.is_doubled else 1
                self._handle_result(amount_rate, "win")
                sleep(0.8)
            elif self.compare(lose, screen):
                amount_rate = 2 if self.is_doubled else 1
                self._handle_result(amount_rate, "lose")
                sleep(0.8)
            elif self.compare(bust, screen):
                _, y = self.compare(bust, screen)
                if y < self.layout["WINDOW_HEIGHT"] / 2:
                    continue
                amount_rate = 2 if self.is_doubled else 1
                self._handle_result(amount_rate, "lose")
                sleep(0.8)
            elif self.compare(draw, screen):
                amount_rate = 2 if self.is_doubled else 1
                self._handle_result(amount_rate, "draw")
                sleep(0.8)
            elif self.compare(blackjack, screen):
                _, y = self.compare(blackjack, screen)
                if y < self.layout["WINDOW_HEIGHT"] / 2:
                    continue
                self._handle_result(1.5, "win")
                sleep(0.8)
            elif self.compare(double, screen):
                # It's the first round - collect all card candidates with position dedup
                candidates = []
                for card_name, card_image in self.card_images.items():
                    res = matchTemplate(screen, card_image, TM_CCOEFF_NORMED)
                    loc = np_where(res >= 0.9)
                    for x, y in zip(*loc[::-1]):
                        val = res[y, x]
                        dup = any(is_close((x, y), (cx, cy), 12) for _, cx, cy, _ in candidates)
                        if dup:
                            existing = [(i, cv) for i, (cn, cx, cy, cv) in enumerate(candidates) if is_close((x, y), (cx, cy), 12)]
                            if existing:
                                idx, old_val = existing[0]
                                if val > old_val:
                                    candidates[idx] = (card_name, x, y, val)
                        else:
                            candidates.append((card_name, x, y, val))
                if not candidates:
                    continue
                first_card = ""
                second_card = ""
                dealer_card = ""
                for card_name, x, y, _ in candidates:
                    self.update_count(card_name)
                    if y < self.layout["WINDOW_HEIGHT"] / 2:
                        if not dealer_card:
                            dealer_card = card_name
                    else:
                        if self.split_round:
                            if self.split_round == 1:
                                first_minn_x, first_maxx_x = (
                                    self.layout["SPLIT_SECOND_GROUP_FIRST_HAND_X"]
                                )
                                second_minn_x, second_maxx_x = (
                                    self.layout["SPLIT_SECOND_GROUP_SECOND_HAND_X"]
                                )
                                if not first_card and first_minn_x < x < first_maxx_x:
                                    first_card = card_name
                                elif not second_card and second_minn_x < x < second_maxx_x:
                                    second_card = card_name
                            elif self.split_round == 2:
                                first_minn_x, first_maxx_x = (
                                    self.layout["SPLIT_FIRST_GROUP_FIRST_HAND_X"]
                                )
                                second_minn_x, second_maxx_x = (
                                    self.layout["SPLIT_FIRST_GROUP_SECOND_HAND_X"]
                                )
                                if not first_card and first_minn_x < x < first_maxx_x:
                                    first_card = card_name
                                elif not second_card and second_minn_x < x < second_maxx_x:
                                    second_card = card_name
                        else:
                            first_minn_x, first_maxx_x = self.layout["FIRST_HAND_X"]
                            second_minn_x, second_maxx_x = self.layout["SECOND_HAND_X"]
                            if not first_card and first_minn_x < x < first_maxx_x:
                                first_card = card_name
                            elif not second_card and second_minn_x < x < second_maxx_x:
                                second_card = card_name
                    if dealer_card and first_card and second_card:
                        break
                if "" in [first_card, second_card, dealer_card]:
                    continue
                card_num1, card_num2 = card_num_from_card_name(
                    first_card
                ), card_num_from_card_name(second_card)
                if card_num1 + card_num2 == 21:
                    self.roundInformUpdated.emit(
                        dealer_card, first_card + "," + second_card, "stand"
                    )
                    self.clickz(self.layout["OP_POS"]["stand"])
                    sleep(0.5)
                    continue
                dealer_card_num_str = card_num_str_from_card_name(dealer_card)
                try:
                    strategy = CHEAT_SHEET[
                        (
                            card_suite_from_two_card_num(card_num1, card_num2),
                            dealer_card_num_str,
                        )
                    ]
                except KeyError:
                    strategy = "stand"
                hand_key = card_suite_from_two_card_num(card_num1, card_num2)
                strategy = self.get_adjusted_strategy(
                    hand_key, dealer_card_num_str, strategy,
                    ["hit", "stand", "double", "split"] if "," in hand_key else ["hit", "stand", "double"]
                )
                if self.learning:
                    self._pending_decisions.append((hand_key, dealer_card_num_str, self.true_count, strategy, self._pending_round))
                if strategy == "double":
                    self.is_doubled = True
                if strategy == "split":
                    if self.split_round > 0:
                        # we can't split again
                        if card_num1 + card_num2 < 12:
                            strategy = "hit"
                        else:
                            strategy = "stand"
                    else:
                        self.split_round = 1
                        self.split_active = True
                        self._pending_round = 1
                        strategy = "split"

                self.roundInformUpdated.emit(
                    dealer_card, first_card + "," + second_card, strategy
                )
                self.emit_count()
                self.clickz(self.layout["OP_POS"][strategy])
            elif self.compare(stand, screen):
                # which means it's the second round and could have mulitple cards
                candidates = []
                for card_name, card_image in self.card_images.items():
                    res = matchTemplate(screen, card_image, TM_CCOEFF_NORMED)
                    loc = np_where(res >= 0.9)
                    for x, y in zip(*loc[::-1]):
                        val = res[y, x]
                        dup = any(is_close((x, y), (cx, cy), 12) for _, cx, cy, _ in candidates)
                        if dup:
                            existing = [(i, cv) for i, (cn, cx, cy, cv) in enumerate(candidates) if is_close((x, y), (cx, cy), 12)]
                            if existing:
                                idx, old_val = existing[0]
                                if val > old_val:
                                    candidates[idx] = (card_name, x, y, val)
                        else:
                            candidates.append((card_name, x, y, val))
                if not candidates:
                    continue
                dealer_card = ""
                total_points = 0
                detected_cards = []
                for card_name, x, y, _ in candidates:
                    if y < self.layout["WINDOW_HEIGHT"] / 2:
                        if not dealer_card:
                            dealer_card = card_name
                        self.update_count(card_name)
                    else:
                        if self.split_active:
                            if self.split_round == 1:
                                min_x, max_x = self.layout["SPLIT_SECOND_GROUP_FIRST_HAND_X"][0], self.layout["SPLIT_SECOND_GROUP_SECOND_HAND_X"][1]
                            else:
                                min_x, max_x = self.layout["SPLIT_FIRST_GROUP_FIRST_HAND_X"][0], self.layout["SPLIT_FIRST_GROUP_SECOND_HAND_X"][1]
                            if min_x <= x <= max_x:
                                total_points += card_num_from_card_name(card_name)
                                self.update_count(card_name)
                                detected_cards.append((card_name, (x, y)))
                                continue
                            else:
                                continue
                        total_points += card_num_from_card_name(card_name)
                        self.update_count(card_name)
                    detected_cards.append((card_name, (x, y)))
                cards = [name for name, pt in detected_cards if name != dealer_card]
                if dealer_card == "" or total_points == 0 or len(cards) < 2:
                    continue
                if total_points >= 21:
                    ace_count = sum(1 for c in cards if len(c) >= 2 and c[1] == "a")
                    for _ in range(ace_count):
                        if total_points <= 21:
                            break
                        total_points -= 10
                dealer_card_num_str = card_num_str_from_card_name(dealer_card)
                hard_sum = 0
                ace_cnt = 0
                for c in cards:
                    if len(c) >= 2 and c[1] == "a":
                        ace_cnt += 1
                        hard_sum += 1
                    else:
                        hard_sum += card_num_from_card_name(c)
                soft_sum = hard_sum
                tmp = ace_cnt
                while tmp and soft_sum + 10 <= 21:
                    soft_sum += 10
                    tmp -= 1
                lookup_key = f"A,{soft_sum - 11}" if soft_sum != hard_sum else str(soft_sum)
                try:
                    strategy = CHEAT_SHEET[
                        (
                            lookup_key,
                            dealer_card_num_str,
                        )
                    ]
                except KeyError:
                    strategy = "stand"
                strategy = self.get_adjusted_strategy(
                    lookup_key, dealer_card_num_str, strategy,
                    ["hit", "stand"]
                )
                if strategy == "double":
                    strategy = "hit"
                if self.learning:
                    self._pending_decisions.append((str(total_points), dealer_card_num_str, self.true_count, strategy, self._pending_round))
                self.roundInformUpdated.emit(dealer_card, ",".join(cards), strategy)
                self.emit_count()
                self.clickz(self.layout["OP_POS"][strategy])
            elif self.compare(bet, screen) is not None:
                if self.card_counting and self.cards_seen > DECKS_IN_SHOE * 52 * 0.75:
                    self.reset_count()
                if self.card_counting and self.wonging and self.true_count < self.wong_tc_threshold:
                    sleep(0.5)
                    continue
                loc = self.compare(bet, screen)
                self.clickz(loc)
                new_bet_key = self.current_bet_key()
                if new_bet_key != self.bet_amount:
                    self.bet_amount = new_bet_key
                    bet = safe_imread(r"image/bet/bet" + self.bet_amount + ".png", 0)
                self.emit_count()

    def stop(self):
        self.terminate()
        self.wait()
