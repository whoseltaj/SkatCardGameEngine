import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL = load_model('player2_model.h5', compile=False)

class Player2AI:
    def __init__(self, player_hands, skat_pile, game_variant, trump_suit):
        self.player_hands = player_hands
        self.skat_pile = skat_pile
        self.game_variant = game_variant
        self.trump_suit = trump_suit

    def generate_training_matrix(self):
        training_matrix = np.zeros((3, len(self.player_hands[0]), len(self.player_hands[0][0])))
        for player_index, hand in enumerate(self.player_hands):
            for card_index, card in enumerate(hand):
                training_matrix[player_index][card_index] = self.card_value(card)
        return training_matrix

    def card_value(self, card):
        rank_values = {'J': 20, 'A': 11, '10': 10, 'K': 4, 'Q': 3, '9': 2, '8': 1, '7': 0}
        return rank_values.get(card.rank, 0)

    def evaluate_bids(self):
        bid_values = [18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 45, 46, 48, 50, 54, 55, 59, 60, 63, 66, 70, 72, 77, 80, 81, 84, 88, 90, 96, 99]
        hand_matrix = self.generate_training_matrix()
        hand_matrix = np.expand_dims(hand_matrix, axis=-1)
        predictions = MODEL.predict(hand_matrix)
        best_bid_index = np.argmax(predictions)
        return bid_values[min(best_bid_index, len(bid_values) - 1)]

    def evaluate_trick_moves(self):
        all_possible_moves = []
        for card in self.player_hands[1]:
            if self.is_valid_move(card):
                all_possible_moves.append(card)
        move_matrices = [self.generate_move_matrix(move) for move in all_possible_moves]
        move_matrices = np.expand_dims(np.array(move_matrices), axis=-1)
        predictions = MODEL.predict(move_matrices)
        best_move_index = np.argmax(predictions)
        return all_possible_moves[best_move_index]

    def is_valid_move(self, card):
        lead_suit = self.get_lead_suit()
        if lead_suit and card.suit != lead_suit and any(c.suit == lead_suit for c in self.player_hands[1]):
            return False
        return True

    def get_lead_suit(self):
        return self.skat_pile[0].suit if self.skat_pile else None

    def generate_move_matrix(self, card):
        matrix = np.zeros((3, len(self.player_hands[0]), len(self.player_hands[0][0])))
        for player_index, hand in enumerate(self.player_hands):
            for card_index, hand_card in enumerate(hand):
                matrix[player_index][card_index] = self.card_value(hand_card)
        return matrix

    def play_turn(self):
        best_move = self.evaluate_trick_moves()
        self.player_hands[1].remove(best_move)
        return best_move

    def make_bid(self):
        return self.evaluate_bids()

    def analyze_trick_outcome(self, trick_cards):
        lead_card = trick_cards[0]
        winning_card = lead_card
        winner_index = 0
        for index, card in enumerate(trick_cards):
            if card.suit == self.trump_suit and winning_card.suit != self.trump_suit:
                winning_card = card
                winner_index = index
            elif card.suit == lead_card.suit and card.rank > winning_card.rank:
                winning_card = card
                winner_index = index
        return winner_index

    def evaluate_next_move(self):
        current_hand = self.player_hands[1]
        move_scores = []
        for card in current_hand:
            simulated_outcome = self.simulate_play(card)
            move_scores.append((card, simulated_outcome))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[0][0]

    def simulate_play(self, card):
        simulated_hand = self.player_hands[1][:]
        simulated_hand.remove(card)
        simulated_matrix = self.generate_training_matrix()
        simulated_matrix = np.expand_dims(simulated_matrix, axis=-1)
        predicted_outcome = MODEL.predict(simulated_matrix)
        return predicted_outcome[0][0]

    def adapt_strategy(self):
        if self.game_variant == "Null":
            self.trump_suit = None
        elif self.game_variant == "Grand":
            self.trump_suit = "J"
        else:
            self.trump_suit = self.determine_best_trump()

    def determine_best_trump(self):
        suit_counts = {suit: 0 for suit in ["Clubs", "Diamonds", "Hearts", "Spades"]}
        for card in self.player_hands[1]:
            suit_counts[card.suit] += 1
        return max(suit_counts, key=suit_counts.get)

    def update_player_hand(self, new_cards):
        self.player_hands[1].extend(new_cards)

    def process_trick_results(self, winner_index):
        if winner_index == 1:
            self.update_player_strategy()
        else:
            self.evaluate_opponent_moves()

    def update_player_strategy(self):
        if len(self.player_hands[1]) < 3:
            self.adapt_strategy()

    def evaluate_opponent_moves(self):
        potential_moves = []
        for card in self.player_hands[0] + self.player_hands[2]:
            if self.is_valid_move(card):
                potential_moves.append(card)
        return potential_moves

    def reset_round(self):
        self.player_hands = [[] for _ in range(3)]
        self.skat_pile = []

    def finalize_game(self):
        total_points = sum(self.card_value(card) for card in self.player_hands[1])
        return total_points

    def log_move(self, card):
        print(f"Player 2 plays {card.rank} of {card.suit}")

    def log_bid(self, bid):
        print(f"Player 2 bids {bid}")

    def display_strategy(self):
        print(f"Current trump suit: {self.trump_suit}")
        print(f"Game variant: {self.game_variant}")

    def save_game_state(self):
        state_data = {
            "player_hands": self.player_hands,
            "skat_pile": self.skat_pile,
            "game_variant": self.game_variant,
            "trump_suit": self.trump_suit
        }
        np.save("game_state.npy", state_data)

    def log_tensor(self, data):
        tensor_data = tf.convert_to_tensor(data)
        print("Tensor Representation:", tensor_data)

    def load_game_state(self):
        state_data = np.load("game_state.npy", allow_pickle=True).item()
        self.player_hands = state_data["player_hands"]
        self.skat_pile = state_data["skat_pile"]
        self.game_variant = state_data["game_variant"]
        self.trump_suit = state_data["trump_suit"]
