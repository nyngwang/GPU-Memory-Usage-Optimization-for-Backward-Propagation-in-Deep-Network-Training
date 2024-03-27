import random
from .algorithm import (
    CONTEXT, clear_cache,
    BF, T_top_down, T_bottom_up, T_bottom_up_mono,
    solve_pivot_selection
)


def example1():
    global CONTEXT
    clear_cache()
    model_len = CONTEXT['MODEL_LEN'] = 7
    print('\nmodel_len: {}'.format(model_len))
    CONTEXT['SEQ_COST'][0] = 7
    CONTEXT['SEQ_COST'][1] = 1 # p
    CONTEXT['SEQ_COST'][2] = 6
    CONTEXT['SEQ_COST'][3] = 3 # p
    CONTEXT['SEQ_COST'][4] = 6
    CONTEXT['SEQ_COST'][5] = 6 # p
    CONTEXT['SEQ_COST'][6] = 7
    # print(BF(model_len, cost))
    # print(T_bottom_up(model_len, cost))
    # print(T_bottom_up_mono(model_len, cost))
    # print()
    # print(solve_PSP(BF))
    # print(solve_PSP(T_bottom_up))
    print(solve_pivot_selection(T_bottom_up_mono))


def example2():
    global CONTEXT
    clear_cache()
    model_len = CONTEXT['MODEL_LEN'] = 6
    print('\nmodel_len: {}'.format(model_len))
    CONTEXT['SEQ_COST'][0] = 5
    CONTEXT['SEQ_COST'][1] = 10
    CONTEXT['SEQ_COST'][2] = 20
    CONTEXT['SEQ_COST'][3] = 5
    CONTEXT['SEQ_COST'][4] = 15
    CONTEXT['SEQ_COST'][5] = 5
    # print(BF(model_len, cost))
    # print(T_bottom_up(model_len, cost))
    # print(T_bottom_up_mono(model_len, cost))
    # print()
    # print(solve_PSP(BF))
    # print(solve_PSP(T_bottom_up))
    print(solve_pivot_selection(T_bottom_up_mono))


def set_random_seq_cost():
    global CONTEXT
    for i in range(CONTEXT['MODEL_LEN']):
        CONTEXT['SEQ_COST'][i] = random.randint(10, 500)


def check_solve_PSP_BF_vs_mono(model_len):
    print('run brute force')
    global CONTEXT
    clear_cache()

    CONTEXT['MODEL_LEN'] = model_len
    print('model len: {}'.format(model_len))

    set_random_seq_cost()
    print('SEQ: {}'.format(CONTEXT['SEQ_COST'][:model_len]))

    cost = sum(CONTEXT['SEQ_COST'][:model_len])
    print('cost: {}'.format(cost))

    # if model_len < 20:
    #     print(BF(model_len, cost, cut=True))
    # print(T_bottom_up(model_len, cost))
    # print(T_bottom_up_mono(model_len, cost))
    # print()
    # if model_len < 20:
    #     print(solve_PSP(BF))
    print(solve_pivot_selection(BF))
    print(solve_pivot_selection(T_bottom_up_mono))


