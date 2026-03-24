from enum import IntEnum


class WordSignal(IntEnum):
    WEAK = 5
    MEDIUM = 10
    STRONG = 15


SIGNAL_WEIGHTS = {"WEAK": 5, "MEDIUM": 10, "STRONG": 15}

LABEL_THRESHOLDS = {"positive_above": 60, "negative_below": 40}


def set_signal_weights(weak: int, medium: int, strong: int) -> None:
    SIGNAL_WEIGHTS["WEAK"] = weak
    SIGNAL_WEIGHTS["MEDIUM"] = medium
    SIGNAL_WEIGHTS["STRONG"] = strong


def set_label_thresholds(positive_above: int, negative_below: int) -> None:
    if negative_below >= positive_above:
        raise ValueError("negative threshold must be less than positive threshold")
    LABEL_THRESHOLDS["positive_above"] = positive_above
    LABEL_THRESHOLDS["negative_below"] = negative_below


NEGATIONS = {"not", "never", "no", "nobody", "nothing", "neither", "nor", "hardly", "barely"}

AMPLIFIERS = {
    "very": 1.5,
    "really": 1.5,
    "extremely": 2.0,
    "absolutely": 2.0,
    "so": 1.3,
    "quite": 1.2,
    "super": 1.5,
    "fr":    1.5,
}

EMOJI_SCORES = {
    "😂": 8,
    "😊": 7,
    "😍": 9,
    "🥰": 9,
    "😄": 8,
    "😁": 7,
    "🎉": 7,
    "❤️": 8,
    "👍": 6,
    "😢": -8,
    "😭": -9,
    "😡": -9,
    "😠": -8,
    "💀": -7,
    "😤": -6,
    "🥲": -4,
    "😞": -7,
    "😔": -6,
}

POSITIVE_WORDS = {
    "okay":     WordSignal.WEAK,
    "fine":     WordSignal.WEAK,
    "chill":    WordSignal.WEAK,
    "hopeful":  WordSignal.WEAK,
    "relaxed":  WordSignal.WEAK,
    "happy":    WordSignal.MEDIUM,
    "great":    WordSignal.MEDIUM,
    "good":     WordSignal.MEDIUM,
    "excited":  WordSignal.MEDIUM,
    "fun":      WordSignal.MEDIUM,
    "proud":    WordSignal.MEDIUM,
    "enjoy":    WordSignal.MEDIUM,
    "grateful": WordSignal.MEDIUM,
    "love":     WordSignal.STRONG,
    "awesome":  WordSignal.STRONG,
    "amazing":  WordSignal.STRONG,
    "wonderful":WordSignal.STRONG,
    "fantastic":WordSignal.STRONG,
    "thriving": WordSignal.STRONG,
    "blessed":  WordSignal.STRONG,
    "joy":      WordSignal.STRONG,
    "hit":       WordSignal.STRONG,
    "different": WordSignal.STRONG,
}

NEGATIVE_WORDS = {
    "tired":        WordSignal.WEAK,
    "boring":       WordSignal.WEAK,
    "bad":          WordSignal.WEAK,
    "sad":          WordSignal.MEDIUM,
    "angry":        WordSignal.MEDIUM,
    "upset":        WordSignal.MEDIUM,
    "stressed":     WordSignal.MEDIUM,
    "worried":      WordSignal.MEDIUM,
    "anxious":      WordSignal.MEDIUM,
    "lonely":       WordSignal.MEDIUM,
    "frustrated":   WordSignal.MEDIUM,
    "disappointed": WordSignal.MEDIUM,
    "terrible":     WordSignal.STRONG,
    "awful":        WordSignal.STRONG,
    "hate":         WordSignal.STRONG,
    "crying":       WordSignal.STRONG,
    "miserable":    WordSignal.STRONG,
    "horrible":     WordSignal.STRONG,
    "exhausted":    WordSignal.STRONG,
    "dread":        WordSignal.STRONG,
}

def add_positive_word(word: str, signal: WordSignal = WordSignal.MEDIUM) -> None:
    word = word.lower()
    NEGATIVE_WORDS.pop(word, None)
    POSITIVE_WORDS[word] = signal


def add_negative_word(word: str, signal: WordSignal = WordSignal.MEDIUM) -> None:
    word = word.lower()
    POSITIVE_WORDS.pop(word, None)
    NEGATIVE_WORDS[word] = signal


def change_word_weight(word: str, signal: WordSignal) -> None:
    word = word.lower()
    if word in POSITIVE_WORDS:
        POSITIVE_WORDS[word] = signal
    elif word in NEGATIVE_WORDS:
        NEGATIVE_WORDS[word] = signal
    else:
        raise KeyError(f"'{word}' not found in POSITIVE_WORDS or NEGATIVE_WORDS")


def add_emoji(emoji: str, score: int) -> None:
    if not -15 <= score <= 15:
        raise ValueError("score must be between -15 and 15")
    EMOJI_SCORES[emoji] = score


def change_emoji_score(emoji: str, score: int) -> None:
    if emoji not in EMOJI_SCORES:
        raise KeyError(f"'{emoji}' not found in EMOJI_SCORES")
    if not -15 <= score <= 15:
        raise ValueError("score must be between -15 and 15")
    EMOJI_SCORES[emoji] = score


def add_amplifier(word: str, multiplier: float) -> None:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    AMPLIFIERS[word.lower()] = multiplier


def add_negation(word: str) -> None:
    NEGATIONS.add(word.lower())


# 100 diverse unlabeled examples for the Active Learning tab
UNLABELED_EXAMPLES = [
    # Clearly positive
    "Just got promoted at work, honestly couldn't be happier right now",
    "Finally finished my thesis after months of work, feels surreal",
    "Had the most amazing brunch with my best friends today 😄",
    "My dog learned a new trick and I am so proud of him",
    "The sunset tonight was absolutely breathtaking 🌅",
    "Got a surprise care package from my mom in the mail",
    "First day at the gym in months and it felt so good",
    "We won the championship game last night 🎉",
    "Just booked flights for my dream vacation, beyond excited",
    "My presentation went really well and the team loved it",
    "Baked cookies from scratch for the first time and they turned out perfect",
    "Got an A on the exam I was most worried about",
    "My little sister graduated today, couldn't be more proud 😊",
    "Random stranger complimented my outfit and made my whole week",
    "Finally paid off my credit card debt 🥳",

    # Clearly negative
    "Failed my driving test for the third time in a row",
    "My landlord is raising my rent again, I can't afford this",
    "Got ghosted after what I thought was a great first date",
    "My flight got cancelled and I missed the whole event",
    "Found out my best friend has been talking behind my back",
    "Woke up with a terrible migraine and it's not getting better",
    "My laptop crashed and I lost three hours of unsaved work",
    "Got a parking ticket right outside my own apartment",
    "My team lead took credit for my project in front of everyone",
    "The restaurant got my order wrong and I'm still hungry 😞",
    "My cat has been sick all week and I'm really worried 😢",
    "Missed my alarm and was late to the most important meeting of the month",
    "My phone screen cracked after dropping it once",
    "Got rejected from every internship I applied to",
    "Had a full blown panic attack in public today, so embarrassing",

    # Neutral / observational
    "Went to the grocery store and grabbed some stuff for the week",
    "The meeting ran about fifteen minutes over schedule",
    "I usually take the bus but decided to walk today",
    "Watched a documentary about penguins last night",
    "Had leftovers from yesterday for dinner",
    "The library closes at nine on Sundays",
    "I switched my phone to dark mode a few weeks ago",
    "Printed out my notes before the lecture",
    "The new coffee shop on my street opened this morning",
    "I usually get there around 8:30",
    "My roommate moved some furniture around this weekend",
    "Set a reminder to call the dentist tomorrow",
    "Took a different route home and it added five minutes",
    "The software update finished installing overnight",
    "I read about forty pages before bed",

    # Mixed / ambiguous
    "Glad the project is over but honestly I learned nothing from it",
    "I love my job but the commute is killing me slowly",
    "Got the apartment I wanted but I'm terrified of living alone",
    "Made it through the week, barely, but made it",
    "The movie was great but the ending ruined the whole thing",
    "Happy to be done with school but kind of scared about what's next",
    "My interview went okay I think, hard to tell honestly",
    "It's my birthday and I'm weirdly sad about it",
    "Finally talked to my ex, it went better than expected but still hurts",
    "Got the job offer but the salary is lower than I was hoping",
    "Finished the race which is amazing but my knee is really sore",
    "Had a good cry tonight, needed it I think",
    "Things are looking up but I don't want to jinx it",
    "Feeling productive but also kind of burnt out at the same time",
    "The party was fun but I'm exhausted and overstimulated now",

    # Sarcastic / ironic
    "Oh great, another all-hands meeting at 8am on a Friday 🙃",
    "Sure, I love being the only one who does any work on the group project",
    "Absolutely thrilling that the WiFi goes down every single day at noon",
    "Nothing like being told last minute that the deadline was actually yesterday",
    "Love how every update somehow makes the app worse",
    "Super convenient that the train was cancelled with zero warning",
    "Oh wow yes please tell me how I should feel about my own life",
    "Great, another email marked urgent that could have been a text",
    "Fantastic timing for the power to go out right before I saved",
    "Always fun when someone schedules a meeting that could have been an email 🙃",

    # School / student life
    "Pulled an all-nighter and still feel like I don't know anything",
    "Group project where I did literally everything",
    "Professor posted the grades and I actually did better than I thought",
    "Spent four hours in the library and barely touched my to-do list",
    "Finally understand recursion after three weeks of confusion",
    "Study group was actually helpful today which never happens",
    "Office hours are so underrated, changed my grade completely",
    "Registration opens in two days and all my classes are already full",
    "Finished reading week with two assignments still untouched",
    "Got an extension so now I have no excuse not to do it right",

    # Health and wellness
    "Ran my first 5K without stopping today",
    "Been drinking more water this week and honestly feel different",
    "Skipped my medication twice this week and really felt it",
    "Finally slept eight hours and forgot what that felt like",
    "My anxiety has been really bad lately, not sure what triggered it",
    "Started meditating in the morning and surprisingly it helps",
    "Went to therapy today and left feeling lighter",
    "Ate nothing but junk food this week and my body is protesting",
    "Took a rest day instead of pushing through and honestly needed it",
    "My back has been bothering me all week from sitting at my desk",

    # Social and relationships
    "Cancelled plans again and I know my friends are getting tired of it",
    "Had a long overdue conversation with my dad and it went better than expected",
    "Made a new friend at work who actually gets my sense of humor",
    "Been feeling really disconnected from everyone around me lately",
    "My sister and I went for a walk and talked for two hours straight",
    "Reached out to someone I fell off with and they responded warmly",
    "Third-wheeled my friends' date night and honestly it was fine",
    "Said no to plans for the first time and didn't feel guilty",
    "My friend remembered something small I mentioned months ago, meant a lot",
    "Spent the whole weekend alone and somehow that was exactly what I needed",
]


def add_sample_post(post: str, label: str) -> None:
    valid_labels = {"positive", "negative", "neutral", "mixed"}
    if label not in valid_labels:
        raise ValueError(f"label must be one of {valid_labels}")
    SAMPLE_POSTS.append(post)
    TRUE_LABELS.append(label)


SAMPLE_POSTS = [
    # --- positive (3) ---
    "I love this class so much",
    "This is absolutely amazing, best day ever",
    "Aced my exam, feeling fantastic 😄",
    # --- negative (3) ---
    "Today was a terrible day",
    "I'm exhausted and miserable, everything is awful",
    "Feeling really anxious and stressed about everything",
    # --- neutral (2) ---
    "Went to the store, came back home",
    "Not much happened today, pretty quiet",
    # --- mixed (2) ---
    "I'm excited but also really nervous",
    "happy about the result but the process was awful",
]

TRUE_LABELS = [
    # --- positive (3) ---
    "positive",
    "positive",
    "positive",
    # --- negative (3) ---
    "negative",
    "negative",
    "negative",
    # --- neutral (2) ---
    "neutral",
    "neutral",
    # --- mixed (2) ---
    "mixed",
    "mixed",
]

