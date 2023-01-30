import math
from string import ascii_uppercase
from enum import Enum


class Constants(Enum):
    PROB_ENGLISH = 0.6
    PROB_SPANISH = 0.4
    LANG_ENGLISH = "English"
    LANG_SPANISH = "Spanish"
    CHAR_A = 'A'
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    LETTER_FILE = "letter.txt"


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return e, s


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                for char in word.upper():
                    if char.isalpha():
                        X[char] = X.get(char, 0) + 1

    for char in ascii_uppercase:
        X[char] = X.get(char, 0)
    return X


def Q1(X):
    print(Constants.Q1.value)
    for char in ascii_uppercase:
        print(char + " " + str(X.get(char, 0)))


def Q2(e, s, X):
    print(Constants.Q2.value)
    x1 = X.get(Constants.CHAR_A.value, 0)
    x1e1 = x1 * math.log(e[0])
    print("%.4f" % x1e1)
    x1s1 = x1 * math.log(s[0])
    print("%.4f" % x1s1)


def functions(language, tuple, X):
    if language == Constants.LANG_ENGLISH.value:
        temp_sum = 0
        for i in range(26):
            temp_sum += X.get(chr(65 + i)) * math.log(tuple[i])
        return math.log(Constants.PROB_ENGLISH.value) + temp_sum
    elif language == Constants.LANG_SPANISH.value:
        temp_sum = 0
        for i in range(26):
            temp_sum += X.get(chr(65 + i)) * math.log(tuple[i])
        return math.log(Constants.PROB_SPANISH.value) + temp_sum


def Q3(e, s, X):
    func_eng = functions(Constants.LANG_ENGLISH.value, e, X)
    func_span = functions(Constants.LANG_SPANISH.value, s, X)
    print(Constants.Q3.value)
    print("%.4f" % func_eng)
    print("%.4f" % func_span)
    return func_eng, func_span


def Q4(func_eng, func_span):
    if func_span - func_eng >= 100:
        prob = 0
    elif func_span - func_eng <= -100:
        prob = 1
    else:
        diff = func_span - func_eng
        exp_diff = math.exp(diff)
        prob = 1 / (1 + exp_diff)
    print(Constants.Q4.value)
    print("%.4f" % prob)


if __name__ == "__main__":
    X = shred(Constants.LETTER_FILE.value)
    Q1(X)
    e, s = get_parameter_vectors()
    Q2(e, s, X)
    func_eng, func_span = Q3(e, s, X)
    Q4(func_eng, func_span)