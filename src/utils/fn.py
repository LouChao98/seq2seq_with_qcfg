from pdb import post_mortem
from typing import List, Tuple

import nltk


def extract_parses(matrix, lengths, kbest=False, inc=0):
    batch = matrix.shape[1] if kbest else matrix.shape[0]
    spans = []
    trees = []
    for b in range(batch):
        if kbest:
            span, tree = extract_parses(
                matrix[:, b], [lengths[b]] * matrix.shape[0], kbest=False, inc=inc
            )
        else:
            span, tree = extract_parse(matrix[b], lengths[b], inc=inc)
        trees.append(tree)
        spans.append(span)
    return spans, trees


def extract_parse(span, length, inc=0):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    N = span.shape[0]
    cover = span.nonzero()
    for i in range(cover.shape[0]):
        w, r, A = cover[i].tolist()
        w = w + inc
        r = r + w
        l = r - w
        spans.append((l, r, A))
        if l != r:
            span = "({} {})".format(tree[l], tree[r])
            tree[r] = tree[l] = span

    return spans, tree[0]


def get_actions(tree, SHIFT=0, REDUCE=1, OPEN="(", CLOSE=")"):
    # input tree in bracket form: ((A B) (C D))
    # output action sequence: S S R S S R R
    actions = []
    tree = tree.strip()
    i = 0
    num_shift = 0
    num_reduce = 0
    left = 0
    right = 0
    while i < len(tree):
        if tree[i] != " " and tree[i] != OPEN and tree[i] != CLOSE:  # terminal
            if tree[i - 1] == OPEN or tree[i - 1] == " ":
                actions.append(SHIFT)
                num_shift += 1
        elif tree[i] == CLOSE:
            actions.append(REDUCE)
            num_reduce += 1
            right += 1
        elif tree[i] == OPEN:
            left += 1
        i += 1
    # assert(num_shift == num_reduce + 1)
    return actions


def get_tree(actions, sent=None, SHIFT=0, REDUCE=1):
    # input action and sent (lists), e.g. S S R S S R R, A B C D
    # output tree ((A B) (C D))
    stack = []
    pointer = 0
    if sent is None:
        sent = list(map(str, range((len(actions) + 1) // 2)))
    #  assert(len(actions) == 2*len(sent) - 1)
    if len(sent) == 1:
        return "(" + sent[0] + ")"
    for action in actions:
        if action == SHIFT:
            word = sent[pointer]
            stack.append(word)
            pointer += 1
        elif action == REDUCE:
            right = stack.pop()
            left = stack.pop()
            stack.append("(" + left + " " + right + ")")
    assert len(stack) == 1
    return stack[-1]


def annotate_snt_with_brackets(
    tokens: List[str], span: List[Tuple[int, int]]
):
    pre_tokens = [0 for _ in range(len(tokens))]
    post_tokens = [0 for _ in range(len(tokens))]

    for l, r, *_ in span:
        if l != r:
            pre_tokens[l] += 1
            post_tokens[r] += 1
    
    output = []
    for pre, token, post in zip(pre_tokens, tokens, post_tokens):
        output_token = []
        if pre > 0:
            output_token.append('(' * pre) 
        output_token.append(token)
        if post > 0:
            output_token.append(')' * post)
        output.append(''.join(output_token))
    return ' '.join(output)
