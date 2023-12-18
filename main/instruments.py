import numpy as np

instruments = {
    1: "Acoustic Grand Piano", #
    2: "Bright Acoustic Piano",
    3: "Electric Grand Piano",
    4: "Honky-tonk Piano",
    5: "Electric Piano 1",
    6: "Electric Piano 2",
    7: "Harpsichord", #
    8: "Clavi",
    9: "Celesta",
    10: "Glockenspiel",
    11: "Music Box",
    12: "Vibraphone",
    13: "Marimba",
    14: "Xylophone",
    15: "Tubular Bells",
    16: "Dulcimer",
    17: "Drawbar Organ",
    18: "Percussive Organ",
    19: "Rock Organ",
    20: "Church Organ",
    21: "Reed Organ",
    22: "Accordion",
    23: "Harmonica",
    24: "Tango Accordion",
    25: "Acoustic Guitar (nylon)",
    26: "Acoustic Guitar (steel)",
    27: "Electric Guitar (jazz)",
    28: "Electric Guitar (clean)",
    29: "Electric Guitar (muted)",
    30: "Overdriven Guitar",
    31: "Distortion Guitar",
    32: "Guitar harmonics",
    33: "Acoustic Bass",
    34: "Electric Bass (finger)",
    35: "Electric Bass (pick)",
    36: "Fretless Bass",
    37: "Slap Bass 1",
    38: "Slap Bass 2",
    39: "Synth Bass 1",
    40: "Synth Bass 2",
    41: "Violin", #
    42: "Viola", #
    43: "Cello", #
    44: "Contrabass", #
    45: "Tremolo Strings",
    46: "Pizzicato Strings",
    47: "Orchestral Harp",
    48: "Timpani",
    49: "String Ensemble 1",
    50: "String Ensemble 2",
    51: "SynthStrings 1",
    52: "SynthStrings 2",
    53: "Choir Aahs",
    54: "Voice Oohs",
    55: "Synth Voice",
    56: "Orchestra Hit",
    57: "Trumpet",
    58: "Trombone",
    59: "Tuba",
    60: "Muted Trumpet",
    61: "French Horn", #
    62: "Brass Section",
    63: "SynthBrass 1",
    64: "SynthBrass 2",
    65: "Soprano Sax",
    66: "Alto Sax",
    67: "Tenor Sax",
    68: "Baritone Sax",
    69: "Oboe", #
    70: "English Horn",
    71: "Bassoon", #
    72: "Clarinet", #
    73: "Piccolo",
    74: "Flute", #
    75: "Recorder",
    76: "Pan Flute",
    77: "Blown Bottle",
    78: "Shakuhachi",
    79: "Whistle",
    80: "Ocarina",
    81: "Lead 1 (square)",
    82: "Lead 2 (sawtooth)",
    83: "Lead 3 (calliope)",
    84: "Lead 4 (chiff)",
    85: "Lead 5 (charang)",
    86: "Lead 6 (voice)",
    87: "Lead 7 (fifths)",
    88: "Lead 8 (bass + lead)",
    89: "Pad 1 (new age)",
    90: "Pad 2 (warm)",
    91: "Pad 3 (polysynth)",
    92: "Pad 4 (choir)",
    93: "Pad 5 (bowed)",
    94: "Pad 6 (metallic)",
    95: "Pad 7 (halo)",
    96: "Pad 8 (sweep)",
    97: "FX 1 (rain)",
    98: "FX 2 (soundtrack)",
    99: "FX 3 (crystal)",
    100: "FX 4 (atmosphere)",
    101: "FX 5 (brightness)",
    102: "FX 6 (goblins)",
    103: "FX 7 (echoes)",
    104: "FX 8 (sci-fi)",
    105: "Sitar",
    106: "Banjo",
    107: "Shamisen",
    108: "Koto",
    109: "Kalimba",
    110: "Bag pipe",
    111: "Fiddle",
    112: "Shanai",
    113: "Tinkle Bell",
    114: "Agogo",
    115: "Steel Drums",
    116: "Woodblock",
    117: "Taiko Drum",
    118: "Melodic Tom",
    119: "Synth Drum",
    120: "Reverse Cymbal",
    121: "Guitar Fret Noise",
    122: "Breath Noise",
    123: "Seashore",
    124: "Bird Tweet",
    125: "Telephone Ring",
    126: "Helicopter",
    127: "Applause",
    128: "Gunshot",
}

instruments_short = {
    1: 1, # 1
    2: 7, # 7
    3: 41, # 41
    4: 42, # 42
    5: 43, # 43
    6: 44, # 44
    7: 61, # 61
    8: 69, # 69
    9: 71, # 71
    10: 72, # 72
    11: 74 # 74
}

instruments_group = {
    1: "Keyboard",
    2: "Organ",
    3: "Accordion",
    4: "Guitar",
    5: "Bass",
    6: "String",
    7: "Ensemble",
    8: "Brass",
    9: "Woodwind",
    10: "Synth",
    11: "Ethnic and percussion",
    12: "Sound effects",
}

instruments_group_short = {
    1: "Keyboard", # 1, 7
    2: "String", # 41, 42, 43, 44
    3: "Brass", # 61
    4: "Woodwind" # 69, 71, 72, 74
}

instruments_arr = np.array(list(instruments.values()))
instruments_group_arr = np.array(list(instruments_group.values()))

#FIXME
instruments_map_arr = {
    instruments_group_arr[0]: instruments_arr[0:16],  # keyboard
    instruments_group_arr[1]: instruments_arr[17:21],  # organ
    instruments_group_arr[2]: instruments_arr[22:24],  # accordion
    instruments_group_arr[3]: instruments_arr[25:32],  # guitar
    instruments_group_arr[4]: instruments_arr[33:40],  # bass
    instruments_group_arr[5]: instruments_arr[40:48],  # string
    instruments_group_arr[6]: instruments_arr[49:56],  # ensemble
    instruments_group_arr[7]: instruments_arr[57:64],  # brass
    instruments_group_arr[8]: instruments_arr[65:80],  # woodwind
    instruments_group_arr[9]: instruments_arr[81:104],  # synth
    instruments_group_arr[10]: instruments_arr[105:120],  # ethnic + percussion
    instruments_group_arr[11]: instruments_arr[121:128],  # sound effects
}

instruments_arr_short = np.array(list(instruments_short.values()))
instruments_group_arr_short = np.array(list(instruments_group_short.values()))

instruments_map_arr_short = {
    instruments_group_arr_short[0]: instruments_arr_short[0:2],  # keyboard
    instruments_group_arr_short[1]: instruments_arr_short[2:6],  # organ
    instruments_group_arr_short[2]: instruments_arr_short[6],  # accordion
    instruments_group_arr_short[3]: instruments_arr_short[7:11],  # guitar
}

#FIXME
instruments_map_arr_alternative = {
    0: np.arange(0, 17),  # keyboard
    1: np.arange(17, 22),  # organ
    2: np.arange(22, 25),  # accordion
    3: np.arange(25, 33),  # guitar
    4: np.arange(33, 41),  # bass
    5: np.arange(40, 49),  # string
    6: np.arange(49, 57),  # ensemble
    7: np.arange(57, 65),  # brass
    8: np.arange(65, 81),  # woodwind
    9: np.arange(81, 105),  # synth
    10: np.arange(105, 121),  # ethnic + percussion
    11: np.arange(121, 129),  # sound effects
}

instruments_map_arr_alternative_short = {
    0: np.arange(0, 2),
    1: np.arange(2,6),
    2: np.array([6]),
    3: np.arange(7,11)
}
