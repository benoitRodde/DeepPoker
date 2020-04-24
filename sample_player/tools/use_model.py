from sample_player.tools.constants import Actions
from sample_player.tools.engine import Engine
from sample_player.tools.models import save_model, push_or_fold_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sample_player.tools.metrics import compute_nash_pusher, compute_nash_caller


def play_model():
    model = push_or_fold_model()
    model.summary()
    engine = Engine()
    num_hands = 5000000
    total_reward = 0

    e = 0.1
    loss = []

    sb_pushed_plt = []
    bb_called_plt = []

    state_sb = np.zeros((16))
    state_bb = np.zeros((16))

    targetQ = np.zeros((2))

    bankroll_sb = [0]
    player1 = True

    replay_state = []
    replay_Q = []
    max_replay = 10000

    sb_pushed = 0
    bb_called = 0

    for i in range(num_hands):
        reward_sb = 0
        reward_bb = 0

        engine.new_hand(starting_stack=np.random.randint(40, 400))

        sb_features = np.concatenate([engine.get_pocket_cards_features(0), np.array(
            [engine.has_suited_pockets(0), 1, engine.starting_stack / 400])]).reshape((1, 16))
        bb_features = np.concatenate([engine.get_pocket_cards_features(1), np.array(
            [engine.has_suited_pockets(1), 0, engine.starting_stack / 400])]).reshape((1, 16))

        replay_state.append(sb_features.reshape((16,)))

        # run model to choose action
        allQ_sb = model.predict(sb_features)
        action_sb = np.argmax(allQ_sb)
        allQ_bb = model.predict(bb_features)
        action_bb = np.argmax(allQ_bb)

        # randomly discover new line
        if (np.random.rand(1) < e):
            action_sb = np.random.randint(0, 2)
            action_bb = np.random.randint(0, 2)

        if action_sb == 1:
            sb_pushed += 1
            engine.play_action(Actions.BET, engine.get_bet_range()[1])

            if action_bb == 1:
                bb_called += 1
                engine.play_action(Actions.CALL, 0)
            else:
                engine.play_action(Actions.FOLD, 0)

        else:
            engine.play_action(Actions.FOLD, 0)

        if engine.winner != -1:
            r = engine.get_sb_won()

        else:
            print('not fichished walla')
        total_reward += r
        bankroll_sb.append(bankroll_sb[-1] + r)

        allQ_sb[0, action_sb] = r
        replay_Q.append(allQ_sb.reshape((2,)))

        # train only if sb played. otherwise positive reward for folding
        if action_sb == 1:
            replay_state.append(bb_features.reshape((16,)))
            allQ_bb[0, action_bb] = -r
            replay_Q.append(allQ_bb.reshape((2,)))

        if (i % 500 == 0 and i > 0):
            print(i)

        if (i % 5000 == 0 and i > 0):
            sb_pushed_pct = int(sb_pushed / (i + 1) * 100)
            sb_pushed_plt.append(sb_pushed_pct)
            bb_called_pct = int(bb_called / (sb_pushed + 1) * 100)
            bb_called_plt.append(bb_called_pct)
            print('\n\nround', i)
            print('Won(SB):', r, '  Total won (SB):', total_reward)
            print('SB pushed ' + str(sb_pushed_pct) + '% , BB called ' + str(bb_called_pct) + '%')
            print('sb_features')
            print(sb_features)
            print('allQ_sb:')
            print(allQ_sb)
            print('\nbb_features')
            print(bb_features)
            print('allQ_bb:')
            print(allQ_bb)

            model.fit(np.stack(replay_state), np.stack(replay_Q), verbose=1, epochs=5)

        if (len(replay_state) > max_replay):
            replay_state = replay_state[50:]
            replay_Q = replay_Q[50:]

    plt.plot(sb_pushed_plt[1:])
    #plt.plot(bb_called_plt[1:])

    call_nash = compute_nash_caller(model)
    push_chart = compute_nash_pusher(model)

    sns.heatmap(push_chart)
    #sns.heatmap(call_nash)
    plt.show()
    save_model(model, 'push_or_fold')

    # model = load('pushfold')

play_model()