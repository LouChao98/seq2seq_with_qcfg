import torch

SHIFT = 0
REDUCE = 1
PT = 2
NT = 3


def get_actions(spans, l):
    spans_set = set([(s[0], s[1]) for s in spans if s[0] < s[1]])
    actions = [SHIFT, SHIFT]
    stack = [(0, 0), (1, 1)]
    ptr = 2
    num_reduce = 0
    while ptr < l:
        if len(stack) >= 2:
            cand_span = (stack[-2][0], stack[-1][1])
        else:
            cand_span = (-1, -1)
        if cand_span in spans_set:
            actions.append(REDUCE)
            stack.pop()
            stack.pop()
            stack.append(cand_span)
            num_reduce += 1
        else:
            actions.append(SHIFT)
            stack.append((ptr, ptr))
            ptr += 1
    while len(actions) < 2 * l - 1:
        actions.append(REDUCE)
    return actions


def inside_with_fully_observed_tree(rules, lengths, spans, nt_add_scores=None):
    # This is similar to BinaryTreeLSTM
    # nt_add_scores use the span indices: bsz x num_tgt_nt_nodes x num_src_nt_nodes
    spans = [[(item[0], item[1] - 1) for item in spans_item] for spans_item in spans]
    results = []

    batched_term = rules["term"]
    batched_rule = rules["rule"]
    batched_root = rules["root"]
    NT = batched_root.shape[1]

    batched_ntnt = batched_rule[..., :NT, :NT]
    batched_ptnt = batched_rule[..., NT:, :NT]
    batched_ntpt = batched_rule[..., :NT, NT:]
    batched_ptpt = batched_rule[..., NT:, NT:]

    for b in range(len(lengths)):
        len_b = lengths[b]
        term = batched_term[b]
        ntnt = batched_ntnt[b]
        ptnt = batched_ptnt[b]
        ntpt = batched_ntpt[b]
        ptpt = batched_ptpt[b]
        root = batched_root[b]

        if nt_add_scores is not None:
            add_score = nt_add_scores[b]
        else:
            add_score = None

        span2id = {span: i for i, span in enumerate(span for span in spans[b] if span[0] != span[1])}
        actions = get_actions(spans[b], len_b)
        assert len_b >= 2
        stack = []

        ptr = 0
        for action in actions:
            if action == SHIFT:
                stack.append((term[ptr], (ptr, ptr)))
                ptr += 1
            else:
                right = stack.pop()
                left = stack.pop()

                left_is_pt = left[1][0] == left[1][1]
                right_is_pt = right[1][0] == right[1][1]

                if left_is_pt and right_is_pt:
                    _rule = ptpt
                elif left_is_pt and not right_is_pt:
                    _rule = ptnt
                elif not left_is_pt and right_is_pt:
                    _rule = ntpt
                else:
                    _rule = ntnt

                head = _rule + left[0][None, :, None] + right[0][None, None, :]
                head = head.flatten(1).logsumexp(1)

                _current_span = (left[1][0], right[1][1])
                if add_score is not None:
                    head = head + add_score[span2id[_current_span]]
                    del span2id[_current_span]  # defense

                stack.append((head, _current_span))

        assert len(stack) == 1
        logpartition = (stack[0][0] + root).logsumexp(0)
        results.append(logpartition)

    return torch.stack(results)


def marginal(rules, lengths, spans, nt_add_scores, constrained=False):
    rules = {
        "term": rules["term"].detach().requires_grad_(),
        "root": rules["root"].detach().requires_grad_(),
        "rule": rules["rule"].detach().requires_grad_(),
    }
    if constrained:
        nt_add_scores = nt_add_scores.detach().requires_grad_()
    else:
        nt_add_scores = torch.zeros_like(nt_add_scores, requires_grad=True)
    inside_with_fully_observed_tree(rules, lengths, spans, nt_add_scores).sum().backward()
    return nt_add_scores.grad.detach()


def cross_entropy(rules, lengths, spans, nt_add_scores):
    rules_c = {
        "term": rules["term"].detach().requires_grad_(),
        "root": rules["root"].detach().requires_grad_(),
        "rule": rules["rule"].detach().requires_grad_(),
    }
    inside_with_fully_observed_tree(rules_c, lengths, spans, nt_add_scores).sum().backward()
    ce = (
        inside_with_fully_observed_tree(rules, lengths, spans, nt_add_scores=None)
        - (rules_c["term"].grad.detach() * rules["term"]).flatten(1).sum(1)
        - (rules_c["rule"].grad.detach() * rules["rule"]).flatten(1).sum(1)
        - (rules_c["root"].grad.detach() * rules["root"]).flatten(1).sum(1)
    )
    return ce
