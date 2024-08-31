import random
import string

vowels = "aeiou"
consonants = "".join(set(string.ascii_lowercase) - set(vowels))


def random_syllable():
    return random.choice(consonants) + random.choice(vowels)


def generate_random_name():
    return "".join([random_syllable() for _ in range(6)])
