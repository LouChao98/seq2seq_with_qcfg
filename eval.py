import argparse
from collections import defaultdict


def read(fname):
    with open(fname) as f:
        data = f.readlines()

    print(f">>> {fname}")
    for i in range(0, len(data), 3):
        print(data[i], end="")
        print(data[i + 1], end="")

        if i == 3:
            break

    output = []
    for raw_snt, labels in zip(data[::3], data[1::3]):
        labels = labels.strip()
        if labels == "":
            output.append([])
            continue
        labels = labels.split("|")
        _labels = []
        for label in labels:
            position, tag = label.split()
            l, r = list(map(int, position.split(",")))
            _labels.append((l, r, tag))
        labels = _labels
        output.append(labels)

    return output


def main(pred, gold, nodup, single):

    pred = read(pred)
    gold = read(gold)
    assert len(pred) == len(gold)

    num_pred = defaultdict(int)
    num_gold = defaultdict(int)
    num_correct = defaultdict(int)

    for p, g in zip(pred, gold):
        p = set(p)  # pred is always uniq

        if nodup:
            g = set(g)  # ace05 contains dup entity (same span and label)
        if single:  # this may introduce randomness. check this whether is a problem
            g = {(l, r): tag for l, r, tag in g}
            g = [(l, r, tag) for (l, r), tag in g.items()]

        gold_by_cat = defaultdict(list)
        for (
            l,
            r,
            tag,
        ) in g:
            gold_by_cat[tag].append((l, r))
        for key, value in gold_by_cat.items():
            num_gold[key] += len(value)

        pred_by_cat = defaultdict(list)
        for l, r, tag in p:
            pred_by_cat[tag].append((l, r))
        for key, value in pred_by_cat.items():
            num_pred[key] += len(value)

        for key, items in pred_by_cat.items():
            for item in items:
                if item in gold_by_cat[key]:
                    num_correct[key] += 1
    total_correct = sum(num_correct.values())
    total_pred = sum(num_pred.values())
    total_gold = sum(num_gold.values())
    P = 100 * total_correct / (total_pred + 1e-9)
    R = 100 * total_correct / (total_gold + 1e-9)
    F = 100 * 2 * total_correct / (total_gold + total_pred + 1e-9)
    print(f"Micro:\tP={P:<6.2f} R={R:<6.2f} F1={F:<6.2f}")
    print("=" * 80)

    ps, rs, fs = [], [], []
    for key in num_gold:
        cat_pred = num_pred[key]
        cat_gold = num_gold[key]
        cat_correct = num_correct[key]
        P = 100 * cat_correct / (cat_pred + 1e-9)
        R = 100 * cat_correct / (cat_gold + 1e-9)
        F = 100 * 2 * cat_correct / (cat_gold + cat_pred + 1e-9)
        print(f"{key}:\tP={P:<6.2f} R={R:<6.2f} F1={F:<6.2f}")
        ps.append(P)
        rs.append(R)
        fs.append(F)

    P = sum(ps) / len(ps)
    R = sum(rs) / len(rs)
    F = sum(fs) / len(fs)
    print(f"Macro:\tP={P:<6.2f} R={R:<6.2f} F1={F:<6.2f}")
    print("=" * 80)

    print(f"Total sentence:    {len(gold)}")
    print(f"Total entities:    {total_gold}")
    print(f"Total predictions: {total_pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred")
    parser.add_argument("gold")
    parser.add_argument("--nodup", action="store_true")
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    main(args.pred, args.gold, args.nodup, args.single)
