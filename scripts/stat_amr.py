from collections import defaultdict

import matplotlib.pyplot as plt
import penman


def stat(src_path):
    cnt_num_en = defaultdict(int)
    cnt_degree = defaultdict(int)
    for graph in penman.iterdecode(open(src_path)):
        cnt_num_en[len([1 for v in graph.reentrancies().values() if v > 1])] += 1
        for v in graph.reentrancies().values():
            cnt_degree[v] += 1

    cnt_num_en = list(cnt_num_en.items())
    cnt_num_en.sort()

    cnt_degree = list(cnt_degree.items())
    cnt_degree.sort()

    fig, axs = plt.subplots(1, 2)
    axs[0].bar([item[0] for item in cnt_num_en], [item[1] for item in cnt_num_en])
    axs[0].set_title("num reentrancy of sentence")
    axs[1].bar([item[0] for item in cnt_degree], [item[1] for item in cnt_degree])
    axs[1].set_title("num of degree of a reentrance")
    fig.savefig("stat.png", figsize=(18, 6))


if __name__ == "__main__":
    stat("data/AMR/tdata_xfm/train.txt.nowiki")
