import random

data_path = "./language_data/word_nums.txt"

newlines = []

intensities = {}

variation_in = 3
variation_out = 0

def pattern_note(type, length, first, magnitude):
    if type == 1:
        # parabolic, up then down
        for i in range(length):
            yield first + round(magnitude-(4*magnitude/((length-1)**2)*(((length-1)/2-(i))**2)))
    if type == 2:
        # linear, down
        for i in range(length):
            yield first + round(-(magnitude*(i/(length-1))))

def pattern_output(type, length):
    if type == 1:
        # linear, decreasing
        start = random.randint(6, 10)
        magnitude = random.randint(3, 5)
        for i in range(length):
            yield round(start-(magnitude*(i/(length-1))))
    if type == 2:
        # linear, increasing
        start = random.randint(1, 5)
        magnitude = random.randint(3, 5)
        for i in range(length):
            yield round(start+(magnitude*(i/(length-1))))

for i in range(100000):
    pattern_type = random.randint(1, 2)
    length = random.randint(10, 20)
    
    first_note = random.randint(15, 45)

    if pattern_type == 1:
        magnitude = random.randint(10, 60-first_note-2)
    elif pattern_type == 2:
        magnitude = random.randint(9, first_note-2)
    notes = " ".join(map(lambda x: str(min(60, max(1, x+((2*random.randint(0,1)-1)*random.randint(0, variation_in))))), pattern_note(pattern_type, length, first_note, magnitude)))
    output = " ".join(map(lambda x: str(min(10, max(1, x+((2*random.randint(0,1)-1)*random.randint(0, variation_out))))), pattern_output(pattern_type, length)))

    newlines.append(notes + "\t" + output + "\n")

f = open("./language_data/notes_numbers_variation.txt", "w")
f.writelines(newlines)
f.close()
