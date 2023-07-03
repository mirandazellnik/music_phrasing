import random

data_path = "./language_data/fra.txt"

newlines = []

intensities = {}

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(10010, len(lines) - 1)]:
    input_text, target_text, c = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.

    target_nums = ""
    for char in input_text:
        x = char.lower()
        if x not in intensities.keys():
            intensities[x] = random.randint(1, 10)
        target_nums += f"{intensities[x]} "
    target_nums = target_nums[:len(target_nums)-1]
    
    newlines.append(input_text + "\t" + target_nums + "\t" + c + "\n")

f = open("./language_data/numbers.txt", "w")
f.writelines(newlines)
f.close()