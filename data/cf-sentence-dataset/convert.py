import json

disciplines = ["Chem", "CL", "Onc", "Psy"]
imrd = ["introduction", "methods", "results", "discussion"]

fo = open("train.jsonl", "a")

for discipline in disciplines:
    for part in imrd:
        # https://iwa2ki.com/FE/
        with open(f"/path/to/NERdepparseLMI/sentence-dataset/train.{discipline}.{part}", "r") as f:
            for line in f:
                cur_line = line.strip()

                cur_label = int(cur_line.split("\t")[0])
                cur_sentence = cur_line.split("\t")[1]

                if part == "methods":
                    cur_label += 11
                elif part == "results":
                    cur_label += 17
                elif part == "discussion":
                    cur_label += 23

                cur_dict = {
                    "text": cur_sentence,
                    "label": cur_label
                }
                
                fo.write(f"{json.dumps(cur_dict)}\n")


fo.close()

class_num = -1
class_str = None

fo = open("dev.jsonl", "a")

for discipline in disciplines:
    for part in imrd:
        # https://iwa2ki.com/FE/
        with open(f"/path/to/NERdepparseLMI/sentence-dataset/eval.{discipline}.{part}", "r") as f:
            for line in f:
                cur_line = line.strip()

                cur_label = int(cur_line.split("\t")[0])
                cur_sentence = cur_line.split("\t")[1]

                if part == "methods":
                    cur_label += 11
                elif part == "results":
                    cur_label += 17
                elif part == "discussion":
                    cur_label += 23

                cur_dict = {
                    "text": cur_sentence,
                    "label": cur_label
                }
                
                fo.write(f"{json.dumps(cur_dict)}\n")



fo.close()


for part in imrd:
    fo = open(f"{part}_all.jsonl", "a")

    for discipline in disciplines:
        # https://iwa2ki.com/FE/
        with open(f"/path/to/NERdepparseLMI/sentence-dataset/train.{discipline}.{part}", "r") as f:
            for line in f:
                cur_line = line.strip()

                cur_label = int(cur_line.split("\t")[0])
                cur_sentence = cur_line.split("\t")[1]

                if part == "methods":
                    cur_label += 11
                elif part == "results":
                    cur_label += 17
                elif part == "discussion":
                    cur_label += 23

                cur_dict = {
                    "text": cur_sentence,
                    "label": cur_label
                }
                
                fo.write(f"{json.dumps(cur_dict)}\n")

        # https://iwa2ki.com/FE/
        with open(f"/path/to/NERdepparseLMI/sentence-dataset/eval.{discipline}.{part}", "r") as f:
            for line in f:
                cur_line = line.strip()

                cur_label = int(cur_line.split("\t")[0])
                cur_sentence = cur_line.split("\t")[1]

                if part == "methods":
                    cur_label += 11
                elif part == "results":
                    cur_label += 17
                elif part == "discussion":
                    cur_label += 23

                cur_dict = {
                    "text": cur_sentence,
                    "label": cur_label
                }
                
                fo.write(f"{json.dumps(cur_dict)}\n")