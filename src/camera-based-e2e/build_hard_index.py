import argparse
import json
import pickle
import os

from protos import e2e_pb2  # same import style you use in loader.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--index_in", type=str, default="index_train.pkl")
    parser.add_argument("--loss_json", type=str, default="scene_loss.json")
    parser.add_argument("--top_k", type=int, default=25000)
    parser.add_argument("--index_out", type=str, default="index_train_hard_25k.pkl")
    args = parser.parse_args()

    # 1) Load loss map and pick top-K names by loss (descending)
    with open(args.loss_json, "r") as f:
        loss_map = json.load(f)  # {name: loss}

    # sort by loss descending
    top = sorted(loss_map.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
    top_names_ordered = [name for name, _ in top]
    top_set = set(top_names_ordered)

    # 2) Load original index tuples
    with open(args.index_in, "rb") as f:
        idx_list = pickle.load(f)

    # 3) Scan index tuples, parse name, keep if in top_set
    # We store found tuples in a dict so we can output in loss-sorted order.
    found = {}

    cur_file = None
    fh = None

    for (filename, start_byte, byte_length) in idx_list:
        if filename != cur_file:
            if fh is not None:
                fh.close()
            fh = open(os.path.join(args.data_dir, filename), "rb")
            cur_file = filename

        fh.seek(start_byte)
        blob = fh.read(byte_length)

        frame = e2e_pb2.E2EDFrame()
        frame.ParseFromString(blob)

        name = frame.frame.context.name
        if name in top_set:
            found[name] = (filename, start_byte, byte_length)

    if fh is not None:
        fh.close()

    # 4) Emit output list in the SAME order as top_names_ordered (highest loss first)
    out_list = [found[n] for n in top_names_ordered if n in found]

    # Sanity check
    missing = [n for n in top_names_ordered if n not in found]
    print(f"Requested top_k={args.top_k}")
    print(f"Found tuples: {len(out_list)}")
    print(f"Missing names: {len(missing)}")
    if missing[:5]:
        print("Example missing:", missing[:5])

    with open(args.index_out, "wb") as f:
        pickle.dump(out_list, f)

    print(f"Wrote hard index: {args.index_out}")


if __name__ == "__main__":
    main()
