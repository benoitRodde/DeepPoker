from pypokerengine.players import BasePokerPlayer
from treys import Card
import numpy as np
from sample_player.tools.models import save_model, load, dqn_model, new_dqn_model
import random


class BetterThanUs(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    uuid = 0
    uuid_adverse = 1
    agent_features = []
    start_stack = None

    def declare_action(self, valid_actions, hole_card, round_state):
        e = random.random()
        exploration_rate = 0.20
        community_card = round_state["community_card"]
        big_blind = round_state['small_blind_amount'] * 2
        my_stack = round_state['seats'][0]['stack']
        opposing_stack = round_state['seats'][1]['stack']
        my_bb_stack = my_stack / big_blind
        opposing_bb_stack = opposing_stack / big_blind

        nb_spade = 0
        nb_heart = 0
        nb_diamond = 0
        nb_clover = 0
        feats, hand_nb_spade, hand_nb_heart, hand_nb_diamond, hand_nb_clover = convert_card_to_treys(community_card)

        agent_feature = np.concatenate([np.array([my_bb_stack, opposing_bb_stack]),
                                        convert_hand_to_treys(hole_card),
                                        np.array([hand_nb_spade, hand_nb_clover, hand_nb_diamond, hand_nb_heart]),
                                        feats[0],
                                        feats[1],
                                        feats[2],
                                        feats[3],
                                        feats[4],
                                        np.array([nb_spade, nb_clover, nb_diamond, nb_heart])
                                        ]).reshape((1, 88))
        if e < exploration_rate:
            print("exploration1")
            self.agent_features.append(agent_feature)
            action = random.choice(valid_actions)["action"]
            if action == "raise":
                action_info = valid_actions[2]
                amount = random.randint(action_info["amount"]["min"], action_info["amount"]["max"])
                if amount == -1:
                    action = "call"
            if action == "call":
                action_info = valid_actions[1]
                amount = action_info["amount"]
            if action == "fold":
                action_info = valid_actions[0]
                amount = action_info["amount"]

            return action, amount
        else:
            #model = new_dqn_model()
            model = load('full_agent_dqn')

            self.agent_features.append(agent_feature)
            model_predict = model.predict(agent_feature)
            raised_v = abs(model_predict.item((0, 2)))
            action_predict = np.argmax(model_predict)
            action_info = convert_int_action_to_str(int(action_predict))
            # On additionne toute les Q value pour avoir le % que represente le raise par raport au autres actions - 33%
            # car si raise -> Qvalue de raise forcement plus grande que 33% des Q values
            print("model_predict1", model_predict)
            percent = raised_v / (abs(model_predict.item((0, 0))) + abs(model_predict.item((0, 1))) + raised_v) * (100 - 33)
            raise_amount = round(percent * my_stack / 100)
            # save_model(model, 'full_agent_dqn')
            return action_to_return(action_info.lower(), valid_actions, raise_amount)

    def receive_game_start_message(self, game_info):

        if game_info["seats"][1]["name"] == "DeepQLearningAgent1":
            self.uuid = game_info["seats"][1]["uuid"]
            self.uuid_adverse = game_info["seats"][0]["uuid"]
            self.start_stack = game_info["seats"][1]["stack"]
        else:
            self.uuid = game_info["seats"][0]["uuid"]
            self.uuid_adverse = game_info["seats"][1]["uuid"]
            self.start_stack = game_info["seats"][0]["stack"]

    def receive_round_start_message(self, round_count, hole_card, seats):
            print("Deep Q agent 1 cards : ", hole_card)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):

        if round_state["seats"][1]["uuid"] == self.uuid:
            end_stack = round_state["seats"][1]["stack"]
        else:
            end_stack = round_state["seats"][0]["stack"]

        value_if_fold = end_stack - self.start_stack
        rewards = []
        lr = 0.70
        replay_Q = []
        actions = []
        replay_state = []

        try:
            if int(winners[0]["uuid"]) == int(self.uuid):
                is_winner = True
            else:
                is_winner = False
        except:
            is_winner = False

        for act in round_state["action_histories"]["preflop"]:
            if act["uuid"] == self.uuid:
                acts, rwds = save_actions(act, is_winner, value_if_fold)
                if len(acts) > 0:
                    actions.append(acts)
                if rwds != -0.5:
                    rewards.append(rwds)
        try:
            for act in round_state["action_histories"]["flop"]:
                if act["uuid"] == self.uuid:
                    acts, rwds = save_actions(act, is_winner, value_if_fold)
                    if len(acts) > 0:
                        actions.append(acts)
                    if rwds != -0.5:
                        rewards.append(rwds)
        except Exception as e:
            pass
        try:
            for act in round_state["action_histories"]["turn"]:
                if act["uuid"] == self.uuid:
                    acts, rwds = save_actions(act, is_winner, value_if_fold)
                    if len(acts) > 0:
                        actions.append(acts)
                    if rwds != -0.5:
                        rewards.append(rwds)
        except Exception as e:
            pass
        try:
            for act in round_state["action_histories"]["river"]:
                if act["uuid"] == self.uuid:
                    acts, rwds = save_actions(act, is_winner, value_if_fold)
                    if len(acts) > 0:
                        actions.append(acts)
                    if rwds != -0.5:
                        rewards.append(rwds)
        except Exception as e:
            pass
        #model = dqn_model()
        model = load('full_agent_dqn')
        for index, action in enumerate(actions):
            int_action = convert_str_action_to_int(action[0])
            replay_state.append(self.agent_features[index].reshape((88,)))
            allQ = model.predict(self.agent_features[index])
            allQ[0, int_action] = allQ[0, int_action] + lr * (rewards[index] - allQ[0, int_action])
            replay_Q.append(allQ.reshape((3,)))
            model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
            model.fit(np.stack(replay_state), np.stack(replay_Q), verbose=1, epochs=5)

        save_model(model, 'full_agent_dqn')
        self.start_stack = end_stack


def setup_ai():
    return BetterThanUs()


def action_to_return(action, valid_actions, raise_amount):
    amount = raise_amount
    if action == "raise":
        action_info = valid_actions[2]
        if action_info['amount']['max'] != -1:
            if raise_amount < action_info["amount"]["min"]:
                amount = action_info["amount"]["min"]
            elif raise_amount > action_info["amount"]["max"]:
                amount = action_info["amount"]["max"]
            else:
                amount = raise_amount
        else:
            action = 'call'
            amount = 0

    elif action == "call":
        action_info = valid_actions[1]
        amount = action_info["amount"]

    elif action == "fold":
        action_info = valid_actions[0]
        amount = action_info["amount"]

    print("DeepQagent1 action :", action, "amount", amount)

    return action, amount  # action returned here is sent to the poker engine


def convert_str_action_to_int(action: str) -> int:
    if action == 'FOLD':
        return 0
    elif action == 'CALL':
        return 1
    elif action == 'RAISE':
        return 2


def convert_int_action_to_str(action: int) -> str:
    if action == 0:
        return 'FOLD'
    elif action == 1:
        return 'CALL'
    elif action == 2:
        return 'RAISE'


def save_actions(act, is_winner, value_if_fold):
    actions = []
    rewards = 0
    if act["action"].lower() == "call":
        actions.append(act["action"])
        if is_winner:
            rewards = act["amount"]
        else:
            rewards = int(act["amount"]) * -1

    elif act["action"].lower() == "fold":
        actions.append(act["action"])
        rewards = value_if_fold

    elif act["action"].lower() == "raise":
        actions.append(act["action"])
        if is_winner:
            rewards = act["amount"]
        else:
            rewards = int(act["amount"]) * -1
    else:
        rewards = -0.5
    return actions, rewards


def convert_card_to_treys(cards):
    convert_cards = []
    hand_nb_spade = 0
    hand_nb_heart = 0
    hand_nb_diamond = 0
    hand_nb_clover = 0
    feats = []

    for card in cards:
        color = card[0]
        num = card[1]
        if color == "C":
            hand_nb_clover += 1
        elif color == "H":
            hand_nb_heart += 1
        elif color == "D":
            hand_nb_diamond += 1
        elif color == "S":
            hand_nb_spade += 1

        convert_cards.append(Card.new(num.upper() + color.lower()))

    for i in [0, 1, 2, 3, 4]:
        feat = np.zeros(13)
        try:
            feat[Card.get_rank_int(convert_cards[i])] = 1
            feats.append(feat)
        except Exception as e:
            feats.append(feat)

    return feats, hand_nb_spade, hand_nb_heart, hand_nb_diamond, hand_nb_clover


def convert_hand_to_treys(hand):
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
