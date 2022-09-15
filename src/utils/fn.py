import logging
from typing import List, Tuple

from nltk.tree import Tree

log = logging.getLogger(__file__)


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
    cover = span.nonzero()
    for i in range(cover.shape[0]):
        w, l, A = cover[i].tolist()
        w = w + inc
        r = l + w
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


def annotate_snt_with_brackets(tokens: List[str], span: List[Tuple[int, int]]):
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
            output_token.append("(" * pre)
        output_token.append(token)
        if post > 0:
            output_token.append(")" * post)
        output.append("".join(output_token))
    return " ".join(output)


def spans2tree(spans: List[Tuple[int, int]], return_mapping=False):
    # mapping = output_span_i -> origin_span_i
    if return_mapping:
        index = list(range(len(spans)))
        index.sort(key=lambda x: (spans[x][0], -spans[x][1]))
        spans = [spans[i] for i in index]
    else:
        spans.sort(key=lambda x: (x[0], -x[1]))
    parent = [i - 1 for i in range(len(spans))]
    endpoint = [-1 for i in range(len(spans))]

    prev_left = spans[0][0]
    prev_left_i = 0

    for i, (s, e, *_) in enumerate(spans[1:], start=1):
        if s == prev_left:
            continue

        endpoint[prev_left_i] = i - 1

        possible_parent_start = prev_left_i
        possible_parent_end = i - 1
        while spans[possible_parent_start][1] < e:
            possible_parent_start = parent[possible_parent_start]
            possible_parent_end = endpoint[possible_parent_start]

        possible_parent_end += 1
        while (possible_parent_end - possible_parent_start) > 1:
            cursor = (possible_parent_start + possible_parent_end) // 2
            v = spans[cursor][1]
            if v < e:
                possible_parent_end = cursor
            elif v == e:
                possible_parent_start = cursor
                break
            else:
                possible_parent_start = cursor

        prev_left = s
        prev_left_i = i
        parent[i] = possible_parent_start
    if return_mapping:
        mapping = list(range(len(spans)))
        mapping.sort(key=lambda x: index[x])
        return spans, parent, mapping
    else:
        return spans, parent


def convert_annotated_str_to_nltk_str(annotated, prefix="x"):
    # convert (I (love nlp)) to (S (NN I) (VP (V love) (NP nlp)))
    # TODO allow escape ( and )

    buffer = []
    left_brace_position = []
    i = 0
    num_token = 0
    while i < len(annotated):
        c = annotated[i]
        i += 1
        if c == " ":
            continue
        elif c == "(":
            left_brace_position.append(len(buffer))
        elif c == ")":
            start = left_brace_position.pop()
            children = buffer[start:]
            tree = Tree(
                (children[0].label()[0], children[-1].label()[1]), buffer[start:]
            )
            buffer[start:] = [tree]
        else:
            j = i + 1
            while j < len(annotated) and annotated[j] not in " ()":
                j += 1
            buffer.append(Tree((num_token, num_token + 1), [annotated[i - 1 : j]]))
            num_token += 1
            i = j
    assert len(buffer) == 1

    def apply_to_tree(tree, func):
        func(tree)
        for subtree in tree:
            if isinstance(subtree, Tree):
                apply_to_tree(subtree, func)

    def relabel_node(tree: Tree):
        s, e = tree._label
        tree._label = f"{prefix}{s}to{e}"

    apply_to_tree(buffer[0], relabel_node)
    return buffer[0]


def report_ids_when_err(func):
    def wrapper(self, batch):
        try:
            return func(self, batch)
        except Exception as e:
            log.error(f"Error at {batch['id']}")
            raise e

    return wrapper
