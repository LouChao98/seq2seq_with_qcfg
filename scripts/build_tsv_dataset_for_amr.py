import penman


def convert(src_path, tgt_path):
    with open(tgt_path, "w") as f:
        for graph in penman.iterdecode(open(src_path)):
            snt = graph.metadata["snt"]
            assert "\t" not in snt
            graph.metadata = {}
            graph_string = penman.encode(graph, indent=None, compact=True)
            f.write(f"{snt}\t{graph_string}\n")


if __name__ == "__main__":
    convert("data/AMR/tdata_xfm/train.txt.nowiki", "data/AMR/train.tsv")
    convert("data/AMR/tdata_xfm/dev.txt.nowiki", "data/AMR/dev.tsv")
    convert("data/AMR/tdata_xfm/test.txt.nowiki", "data/AMR/test.tsv")
