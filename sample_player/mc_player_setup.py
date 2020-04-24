import json

import numpy as np
from pypokerengine.players import BasePokerPlayer
from treys import Card
import os
from sample_player.tools.models import load
from sample_player.tools.utils import hand_strength_estimation, hand_strength_estimation_with_time


def save_board_in_tab_and_return_equity(nb_players, saved_tab, hole_card, tmp_hole_card, community_card):
    hand_equity = 0
    string_community = ' '.join(map(str, community_card))

    if len(tmp_hole_card) > 1:
        hand = tmp_hole_card
        string_hand = ' '.join(map(str, hand))
    else:
        hand = hole_card
        string_hand = ' '.join(map(str, hand))
    found = False
    for board, equity in saved_tab[string_hand].items():
        if len(set(community_card).intersection(set(board.split()))) == len(community_card):
            hand_equity = equity
            found = True

    if found is False:
        a = set(hole_card).intersection(set(board.split()))
        # hand_equity = hand_strength_estimation(100000, nb_players, hand, community_card)
        hand_equity = hand_strength_estimation_with_time(5, nb_players, hand, community_card)
        saved_tab[string_hand].update({string_community:  hand_equity})

    return hand_equity, saved_tab


def get_state_equity(nb_players, hole_card, tmp_hole_card, community_card=None):
    string_hole_card = ' '.join(map(str, hole_card))
    label_pre_flop = "pre_flop"
    saved_tab = grab_state_equity()
    hand_equity = 0
    if community_card is not None and len(community_card) > 0:
        hand_equity, saved_tab = save_board_in_tab_and_return_equity(nb_players, saved_tab, hole_card, tmp_hole_card, community_card)
    else:
        try:
            hand_equity = saved_tab[string_hole_card][label_pre_flop]
        except Exception as e:
            found = False
            for key, board in saved_tab.items():
                if len(set(hole_card).intersection(set(key.split()))) == 2:
                    hand_equity = board['pre_flop']
                    tmp_hole_card = key.split(" ")
                    found = True
                    break
            if found is False:
                # hand_equity = hand_strength_estimation(100000, nb_players, hole_card, community_card)
                hand_equity = hand_strength_estimation_with_time(5, nb_players, hole_card, community_card)
                saved_tab[string_hole_card] = {"pre_flop":  hand_equity}

    save_state_equity(saved_tab)
    return hand_equity, tmp_hole_card


class MCPlayer(BasePokerPlayer):
    uuid = 0
    uuid_adverse = 1
    tmp_hole_card = []
    my_position = 0

    def receive_game_start_message(self, game_info):
        if game_info["seats"][0]["name"] == "MonteCarloAgent":
            self.uuid = game_info["seats"][0]["uuid"]
            self.uuid_adverse = game_info["seats"][1]["uuid"]
        else:
            self.uuid = game_info["seats"][1]["uuid"]
            self.uuid_adverse = game_info["seats"][0]["uuid"]

    def receive_round_start_message(self, round_count, hole_card, seats):
        print("MonteCarlo cards : ", hole_card)
        if seats[0]["name"] == "MonteCarloAgent":
            self.my_position = 0
        else:
            self.my_position = 1
        self.tmp_hole_card = []

    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state["street"]
        big_blind = round_state['small_blind_amount'] * 2
        my_stack = round_state['seats'][0]['stack']
        opposing_stack = round_state['seats'][1]['stack']
        nb_players = 2
        pot = round_state['pot']['main']['amount']
        my_bb_stack = my_stack / big_blind
        opposing_bb_stack = opposing_stack / big_blind
        action = 'call'
        raise_amount = 0
        treys_hand = convert_card_to_treys(hole_card)
        suited_hand = is_suited_hand(hole_card)

        # push or fold if short stack or or if opponent is short stack
        if my_bb_stack < 20 or opposing_bb_stack < 20 or my_bb_stack > (opposing_bb_stack * 20 + pot/big_blind):
            model = load("sample_player/tools/push_or_fold")
            print("Push or fold")
            # model.summary()
            # Si premier à parler
            if round_state['small_blind_pos'] == self.my_position:
                print("premier de parole")
                my_features = np.concatenate((treys_hand, np.array(
                    [suited_hand, 1, my_bb_stack]))).reshape((1, 16))
                allQ_sb = model.predict(my_features)
                action_sb = np.argmax(allQ_sb)
                # fold
                if action_sb == 0:
                    action = 'fold'
                    raise_amount = 0

                # shove
                elif action_sb == 1:
                    action = 'raise'
                    raise_amount = valid_actions[2]['amount']['max']

            # Si on est second à parler
            else:
                print("deuxieme de parole")
                bb_features = np.concatenate((treys_hand, np.array(
                    [suited_hand, 0, my_bb_stack]))).reshape((1, 16))
                allQ_bb = model.predict(bb_features)
                action_bb = np.argmax(allQ_bb)
                # Si action fold
                if action_bb == 0:
                    # Check si possible
                    if (valid_actions[1]['amount'] == big_blind and street == "preflop") or valid_actions[1]['amount'] == 0 and street != "preflop":
                        action = 'call'
                        raise_amount = 0
                    # Sinon fold
                    else:
                        action = 'fold'
                        raise_amount = 0
                # Si pas fold => partir à tapis
                elif action_bb == 1:
                    action = 'raise'
                    raise_amount = valid_actions[2]['amount']['max']

        # not push or fold
        else:
            # Calcul de l'équité de la main (use monte carlo)
            hand_equity, self.tmp_hole_card = get_state_equity(nb_players, hole_card, self.tmp_hole_card, round_state['community_card'])
            # Si premier relanceur ( call amount = 0 )
            if valid_actions[1]['amount'] < 2 * big_blind:
                # Si équité > 65
                if hand_equity > 0.5600:
                    # relance un montant fixe 1/2 de la taille du pot
                    action = 'raise'
                    if pot > big_blind * 3:
                        raise_amount = round(pot / 2, 0)
                    else:
                        raise_amount = round(big_blind * 3, 0)

                # si équité < 35
                elif hand_equity < 0.3500:
                    print("bluff low equity")
                    # relance meme montant fixe + haut ( bluff ) 1/2 du pot
                    action = 'raise'
                    raise_amount = round(pot / 2, 0)
                # sinon call 0
                else:
                    action = 'call'
                    raise_amount = 0
            # Si il y relance avant
            else:
                # calcul de la cote
                action_info = valid_actions[1]
                amount = action_info["amount"]
                cote = amount / pot
                # Si côte au dessus d'un stade call n'importe quel mise
                if hand_equity > 0.8750 and round_state["street"] == "river":
                    if valid_actions[2]['amount']['max'] != -1:
                        action = 'raise'
                        raise_amount = valid_actions[2]['amount']['max']
                    else:
                        action = 'call'
                        raise_amount = 0

                elif hand_equity > 0.6900:
                    action = 'call'
                    raise_amount = 0
                elif hand_equity > 0.5600 and valid_actions[1]['amount'] < 6 * big_blind:
                    action = 'call'
                    raise_amount = 0

                elif hand_equity > cote:
                    action = 'call'
                    raise_amount = 0
                # Sinon calcul de la cote pour savoir si call ou fold

                else:
                    action = 'fold'
                    raise_amount = 0
        return action_to_return(action, valid_actions, raise_amount)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def action_to_return(action, valid_actions, raise_amount):
    if action == "raise":
        action_info = valid_actions[2]
        if raise_amount < action_info["amount"]["min"]:
            amount = action_info["amount"]["min"]
        elif raise_amount > action_info["amount"]["max"]:
            amount = action_info["amount"]["max"]
        else:
            amount = raise_amount
    if action == "call":
        action_info = valid_actions[1]
        amount = action_info["amount"]
    if action == "fold":
        action_info = valid_actions[0]
        amount = action_info["amount"]
    print("MonteCarloAgent action :", action, "amount", amount)
    return action, amount  # action returned here is sent to the poker engine


def save_state_equity(saved_tab):
    if os.getcwd() == '/home/benoit/Documents/Projet_IA/Perso/Poker/PokerAI':
        with open("sample_player/tools/hu_hands_equity.json", "w") as file:
            json.dump(saved_tab, file)
    else:
        with open("tools/hu_hands_equity.json", "w") as file:
            json.dump(saved_tab, file)


def grab_state_equity():
    if os.getcwd() == '/home/benoit/Documents/Projet_IA/Perso/Poker/PokerAI':
        with open("sample_player/tools/hu_hands_equity.json", "r") as file:
            saved_tab = json.load(file)
        return saved_tab
    else:
        with open("tools/hu_hands_equity.json", "r") as file:
            saved_tab = json.load(file)
        return saved_tab


def convert_card_to_treys(hand):
    num = []
    color = []

    for card in hand:
        color.append(card[0])
        num.append(card[1])

    card1 = Card.new(num[0].upper() + color[0].lower())
    card2 = Card.new(num[1].upper() + color[1].lower())

    feat = np.zeros(13)
    for c in [card1, card2]:
        feat[Card.get_rank_int(c)] = 1
    return feat


def is_suited_hand(hand):
    color = []

    for card in hand:
        color.append(card[0])

    if color[0] == color[1]:
        return True
    else:
        return False


def setup_ai():
    return MCPlayer()
