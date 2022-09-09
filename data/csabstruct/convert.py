import pandas as pd
import json

for file_type in ["train", "dev", "test"]:
    tmp_txt = ""

    with open(f"original/{file_type}.jsonl", "r") as f:
        for line in f:
            dic = json.loads(line)

            assert len(dic["sentences"]) == len(dic["labels"])
            n = len(dic["sentences"])

            for i in range(n):
                tmp_dic = dict()
                tmp_dic["text"] = dic["sentences"][i]
                tmp_dic["label"] = dic["labels"][i]
                tmp_txt += f"{json.dumps(tmp_dic)}\n"

    with open(f"{file_type}.jsonl", "w") as f:
        f.write(tmp_txt)