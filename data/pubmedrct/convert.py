import json

with open("original/test.txt", "r") as f:
    tmp_txt = ""

    for line in f:
        cur_line = line.strip()

        if cur_line == "" or cur_line.startswith("###"):
            continue

        split_line = cur_line.split("\t")
        tmp_dic = {
            "text": split_line[1],
            "label": split_line[0]
        }
        tmp_txt += f"{json.dumps(tmp_dic)}\n"

    with open(f"test.jsonl", "w") as fo:
        fo.write(tmp_txt)