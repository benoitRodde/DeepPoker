from mc_player_setup import get_state_equity

suits1 = ['H', 'C', 'S', 'D']

indexes1 = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


def train_state_equity():
    """ MY HAND """
    # Carte 1
    for i1 in indexes1:
        for s1 in suits1:
            indexes2 = indexes1.copy()
            suites2 = suits1.copy()
            # Carte 2
            for i2 in indexes2:
                for s2 in suites2:
                    card1 = [s1+i1]
                    card2 = [s2+i2]
                    if card1 == card2:
                        continue
                    hole_card = [s1+i1, s2+i2]
                    tmp_hole_card = []
                    indexes3 = indexes2.copy()
                    suites3 = suites2.copy()
                    # LEARN
                    print(hole_card)
                    _, tmp_hole_card = get_state_equity(2, hole_card, tmp_hole_card, [])
                    """ FLOP """
                    # Carte 3
                    for i3 in indexes3:
                        for s3 in suites3:
                            burn_card = hole_card
                            card3 = [s3+i3]
                            if len(set(card3).intersection(set(burn_card))) > 0:
                                continue
                            indexes4 = indexes3.copy()
                            suites4 = suites3.copy()
                            # Carte 4
                            for i4 in indexes4:
                                for s4 in suites4:
                                    card4 = [s4+i4]
                                    burn_card = card3 + hole_card
                                    if len(set(card4).intersection(set(burn_card))) > 0:
                                        continue
                                    indexes5 = indexes4.copy()
                                    suites5 = suites4.copy()
                                    # Carte 5
                                    for i5 in indexes5:
                                        for s5 in suites5:
                                            card5 = [s5+i5]
                                            burn_card = hole_card + card3 + card4
                                            if len(set(card5).intersection(set(burn_card))) > 0:
                                                continue
                                            community_card = card3 + card4 + card5
                                            indexes6 = indexes5.copy()
                                            suites6 = suites5.copy()
                                            get_state_equity(2, hole_card, tmp_hole_card, community_card)
                                            # Carte 6
                                            """ TURN """
                                            for i6 in indexes6:
                                                for s6 in suites6:
                                                    card6 = [s6+i6]
                                                    burn_card = hole_card + card3 + card4 + card5 + hole_card
                                                    if len(set(card6).intersection(set(burn_card))) > 0:
                                                        continue
                                                    community_card = card3 + card4 + card5 + card6
                                                    indexes7 = indexes6.copy()
                                                    suites7 = suites6.copy()
                                                    _, tmp_hole_card = get_state_equity(2, hole_card, tmp_hole_card, community_card)
                                                    # Carte 7
                                                    """ RIVER """
                                                    for i7 in indexes7:
                                                        for s7 in suites7:
                                                            card7 = [s7+i7]
                                                            burn_card = hole_card + card3 + card4 + card5 + card6
                                                            if len(set(card7).intersection(set(burn_card))) > 0:
                                                                continue
                                                            community_card = card3 + card4 + card5 + card6 + card7
                                                            _, tmp_hole_card = get_state_equity(2, hole_card, tmp_hole_card, community_card)


train_state_equity()
