"""
Semi-synthetic vocabulary lists for counterfactual dataset generation.

Format: [source-concept]_[common-concept]

All aligned lists (eng_objs/french_objs/german_objs, and the four gendered
lists) are index-aligned: position i refers to the same underlying concept.
"""

import json
from pathlib import Path


_CORR_OCCUPATION_PATH = Path(__file__).with_name("vocab") / "corr_occupations.json"
with _CORR_OCCUPATION_PATH.open(encoding="utf-8") as _corr_occupation_file:
    corr_occupation_rows = json.load(_corr_occupation_file)

corr_occ_eng = [row["eng"] for row in corr_occupation_rows]
corr_occ_french = [row["french"] for row in corr_occupation_rows]
corr_occ_german = [row["german"] for row in corr_occupation_rows]

eng_objs = [
    "Friday",
    "Chair",
    "Book",
    "Pen",
    "Monday",
    "Ring",
    "Computer",
    "Kid",
    "Car",
    "Bottle",
    "Cup",
    "Bag",
    "Key",
    "Door",
    "Window",
    "fish",
    "Pillow",
    "Blanket",
    "Sunday",
    "Television",
    "Clock",
    "Watch",
    "Wallet",
    "Shoe",
    "Dog",
    "Shirt",
    "Pants",
    "Hat",
    "Umbrella",
    "Glasses",
    "Spoon",
    "Fork",
    "Knife",
    "Plate",
    "Bowl",
    "Toothbrush",
    "Toothpaste",
    "Towel",
    "Soap",
    "Shampoo",
    "Mirror",
    "Comb",
    "Brush",
    "Candle",
    "Saturday",
    "Flower",
    "Bicycle",
    "Ball",
    "Toy",
    "Pillar",
    "Microwave",
    "Oven",
    "Refrigerator",
    "Stove",
    "Sink",
    "Toilet",
    "Bathtub",
    "Shower",
    "Desk",
    "Notebook",
    "Calendar",
    "Stamp",
    "Coin",
    "Money",
    "Basket",
    "Remote",
    "Headphones",
    "Speaker",
    "Lantern",
    "Printer",
    "Mouse",
    "Keyboard",
    "Monitor",
    "Laptop",
    "Curtain",
    "Suitcase",
    "Map",
    "Vase",
    "Camera",
    "Painting",
    "Cushion",
    "Hammer",
    "Screwdriver",
    "Wrench",
    "Drill",
    "Nail",
    "Screw",
    "Tape",
    "Glue",
    "Scissors",
    "Rope",
    "Lighter",
    "Ladder",
    "Bucket",
    "Mop",
    "Broom",
    "Trash Can",
    "Thermos",
    "Vacuum",
    "Fan",
]

french_objs = [
    "Vendredi",
    "Chaise",
    "Livre",
    "Stylo",
    "Lundi",
    "Bague",
    "Ordinateur",
    "Enfant",
    "Voiture",
    "Bouteille",
    "Tasse",
    "Sac",
    "Clé",
    "Porte",
    "Fenêtre",
    "poisson",
    "Oreiller",
    "Couverture",
    "Dimanche",
    "Télévision",
    "Horloge",
    "Montre",
    "Portefeuille",
    "Chaussure",
    "Chien",
    "Chemise",
    "Pantalon",
    "Chapeau",
    "Parapluie",
    "Lunettes",
    "Cuillère",
    "Fourchette",
    "Couteau",
    "Assiette",
    "Bol",
    "Brosse à dents",
    "Dentifrice",
    "Serviette",
    "Savon",
    "Shampoing",
    "Miroir",
    "Peigne",
    "Brosse",
    "Bougie",
    "Samedi",
    "Fleur",
    "Vélo",
    "Balle",
    "Jouet",
    "Pilier",
    "Micro-ondes",
    "Four",
    "Réfrigérateur",
    "Cuisinière",
    "Évier",
    "Toilettes",
    "Baignoire",
    "Douche",
    "Bureau",
    "Cahier",
    "Calendrier",
    "Timbre",
    "Pièce",
    "Argent",
    "Panier",
    "Télécommande",
    "Casque",
    "Haut-parleur",
    "Lanterne",
    "Imprimante",
    "Souris",
    "Clavier",
    "Écran",
    "Ordinateur portable",
    "Rideau",
    "Valise",
    "Carte",
    "Vase",
    "Appareil photo",
    "Peinture",
    "Coussin",
    "Marteau",
    "Tournevis",
    "Clé à molette",
    "Perceuse",
    "Clou",
    "Vis",
    "Ruban adhésif",
    "Colle",
    "Ciseaux",
    "Corde",
    "Briquet",
    "Échelle",
    "Seau",
    "Serpillière",
    "Balai",
    "Poubelle",
    "Thermos",
    "Aspirateur",
    "Ventilateur",
]

german_objs = [
    "Freitag",  # Friday
    "Stuhl",  # Chair
    "Buch",  # Book
    "Stift",  # Pen
    "Montag",  # Monday
    "Ring",  # Ring
    "Computer",  # Computer
    "Kind",  # Child
    "Auto",  # Car
    "Flasche",  # Bottle
    "Tasse",  # Cup
    "Tasche",  # Bag
    "Schlüssel",  # Key
    "Tür",  # Door
    "Fenster",  # Window
    "Fisch",  # fish
    "Kissen",  # Pillow
    "Decke",  # Blanket
    "Sonntag",  # Sunday
    "Fernseher",  # Television
    "Uhr",  # Clock
    "Armbanduhr",  # Watch
    "Geldbörse",  # Wallet
    "Schuh",  # Shoe
    "Hund",  # Dog
    "Hemd",  # Shirt
    "Hose",  # Pants
    "Hut",  # Hat
    "Regenschirm",  # Umbrella
    "Brille",  # Glasses
    "Löffel",  # Spoon
    "Gabel",  # Fork
    "Messer",  # Knife
    "Teller",  # Plate
    "Schüssel",  # Bowl
    "Zahnbürste",  # Toothbrush
    "Zahnpasta",  # Toothpaste
    "Handtuch",  # Towel
    "Seife",  # Soap
    "Shampoo",  # Shampoo
    "Spiegel",  # Mirror
    "Kamm",  # Comb
    "Bürste",  # Brush
    "Kerze",  # Candle
    "Samstag",  # Saturday
    "Blume",  # Flower
    "Fahrrad",  # Bicycle
    "Ball",  # Ball
    "Spielzeug",  # Toy
    "Säule",  # Pillar
    "Mikrowelle",  # Microwave
    "Ofen",  # Oven
    "Kühlschrank",  # Refrigerator
    "Herd",  # Stove
    "Spüle",  # Sink
    "Toilette",  # Toilet
    "Badewanne",  # Bathtub
    "Dusche",  # Shower
    "Schreibtisch",  # Desk
    "Notizbuch",  # Notebook
    "Kalender",  # Calendar
    "Briefmarke",  # Stamp
    "Münze",  # Coin
    "Geld",  # Money
    "Korb",  # Basket
    "Fernbedienung",  # Remote
    "Kopfhörer",  # Headphones
    "Lautsprecher",  # Speaker
    "Laterne",  # Lantern
    "Drucker",  # Printer
    "Maus",  # Mouse
    "Tastatur",  # Keyboard
    "Monitor",  # Monitor
    "Laptop",  # Laptop
    "Vorhang",  # Curtain
    "Koffer",  # Suitcase
    "Karte",  # Map
    "Vase",  # Vase
    "Kamera",  # Camera
    "Gemälde",  # Painting
    "Kissen",  # Cushion
    "Hammer",  # Hammer
    "Schraubenzieher",  # Screwdriver
    "Schraubenschlüssel",  # Wrench
    "Bohrer",  # Drill
    "Nagel",  # Nail
    "Schraube",  # Screw
    "Klebeband",  # Tape
    "Kleber",  # Glue
    "Schere",  # Scissors
    "Seil",  # Rope
    "Feuerzeug",  # Lighter
    "Leiter",  # Ladder
    "Eimer",  # Bucket
    "Mopp",  # Mop
    "Besen",  # Broom
    "Mülleimer",  # Trash Can
    "Thermoskanne",  # Thermos
    "Staubsauger",  # Vacuum
    "Ventilator",  # Fan
]

eng_masc = [
    "actor",
    "batman",
    "boar",
    "boy",
    "brother",
    "buck",
    "bull",
    "businessman",
    "chairman",
    "dad",
    "daddy",
    "duke",
    "emperor",
    "father",
    "fisherman",
    "fox",
    "gentleman",
    "god",
    "grandfather",
    "grandpa",
    "grandson",
    "groom",
    "he",
    "headmaster",
    "heir",
    "hero",
    "husband",
    "king",
    "lion",
    "man",
    "manager",
    "men",
    "mister",
    "murderer",
    "nephew",
    "poet",
    "policeman",
    "prince",
    "ram",
    "rooster",
    "sculptor",
    "sir",
    "son",
    "stallion",
    "stepfather",
    "superman",
    "tiger",
    "uncle",
    "servant",
    "waiter",
    "webmaster",
    "wizard",
    "landlord",
    "host",
    "priest",
    "abbot",
    "count",
    "monk",
    "lad",
    "male",
    "peacock",
    "steward",
    "master",
    "widower",
    "sultan",
    "earl",
    "son-in-law",
    "fiancé",
    "bridegroom",
    "bachelor",
    "milkman",
    "manservant",
    "baron",
    "salesman",
    "conductor",
    "godfather",
    "boyfriend",
    "shepherd",
    "papa",
    "stepson",
    "godson",
    "testator",
    "executor",
    "hunter",
    "mayor",
    "peer",
    "singer",
    "tempter",
    "adventurer",
    "proprietor",
    "giant",
    "marquess",
    "czar",
    "author",
    "enchanter",
    "tailor",
    "prophet",
]
# 97

eng_fem = [
    "actress",
    "batwoman",
    "sow",
    "girl",
    "sister",
    "doe",
    "cow",
    "businesswoman",
    "chairwoman",
    "mom",
    "mommy",
    "duchess",
    "empress",
    "mother",
    "fisherwoman",
    "vixen",
    "lady",
    "goddess",
    "grandmother",
    "grandma",
    "granddaughter",
    "bride",
    "she",
    "headmistress",
    "heiress",
    "heroine",
    "wife",
    "queen",
    "lioness",
    "woman",
    "manageress",
    "women",
    "miss",
    "murderess",
    "niece",
    "poetess",
    "policewoman",
    "princess",
    "ewe",
    "hen",
    "sculptress",
    "madam",
    "daughter",
    "mare",
    "stepmother",
    "superwoman",
    "tigress",
    "aunt",
    "maid",
    "waitress",
    "webmistress",
    "witch",
    "landlady",
    "hostess",
    "priestess",
    "abbess",
    "countess",
    "nun",
    "lass",
    "female",
    "peahen",
    "stewardess",
    "mistress",
    "widow",
    "sultana",
    "viscountess",
    "daughter-in-law",
    "fiancée",
    "matron",
    "spinster",
    "milkmaid",
    "maidservant",
    "baroness",
    "saleswoman",
    "conductress",
    "godmother",
    "girlfriend",
    "shepherdess",
    "mama",
    "stepdaughter",
    "goddaughter",
    "testatrix",
    "executrix",
    "huntress",
    "mayoress",
    "peeress",
    "songstress",
    "temptress",
    "adventuress",
    "proprietress",
    "giantess",
    "marchioness",
    "czarina",
    "authoress",
    "enchantress",
    "seamstress",
    "prophetess",
]
# 97

french_masc = [
    "acteur",
    "batman",
    "sanglier",
    "garçon",
    "frère",
    "cerf",
    "taureau",
    "homme d'affaires",
    "président",
    "papa",
    "papa",
    "duc",
    "empereur",
    "père",
    "pêcheur",
    "renard",
    "gentleman",
    "dieu",
    "grand-père",
    "papi",
    "petit-fils",
    "marié",
    "il",
    "directeur",
    "héritier",
    "héros",
    "mari",
    "roi",
    "lion",
    "homme",
    "gérant",
    "hommes",
    "monsieur",
    "meurtrier",
    "neveu",
    "poète",
    "policier",
    "prince",
    "bélier",
    "coq",
    "sculpteur",
    "monsieur",
    "fils",
    "étalon",
    "beau-père",
    "superman",
    "tigre",
    "oncle",
    "domestique",
    "serveur",
    "webmaster",
    "sorcier",
    "propriétaire",
    "hôte",
    "prêtre",
    "abbé",
    "comte",
    "moine",
    "garçon",
    "mâle",
    "paon",
    "steward",
    "maître",
    "veuf",
    "sultan",
    "vicomte",
    "gendre",
    "fiancé",
    "marié",
    "célibataire",
    "laitier",
    "domestique",
    "baron",
    "vendeur",
    "chef d'orchestre",
    "parrain",
    "petit ami",
    "berger",
    "papa",
    "beau-fils",
    "filleul",
    "testateur",
    "exécuteur",
    "chasseur",
    "maire",
    "pair",
    "chanteur",
    "tentateur",
    "aventurier",
    "propriétaire",
    "géant",
    "marquis",
    "tsar",
    "auteur",
    "enchanteur",
    "tailleur",
    "prophète",
]
# 97

french_fem = [
    "actrice",
    "batwoman",
    "truie",
    "fille",
    "sœur",
    "biche",
    "vache",
    "femme d'affaires",
    "présidente",
    "maman",
    "maman",
    "duchesse",
    "impératrice",
    "mère",
    "pêcheuse",
    "renarde",
    "dame",
    "déesse",
    "grand-mère",
    "mamie",
    "petite-fille",
    "mariée",
    "elle",
    "directrice",
    "héritière",
    "héroïne",
    "épouse",
    "reine",
    "lionne",
    "femme",
    "gérante",
    "femmes",
    "mademoiselle",
    "meurtrière",
    "nièce",
    "poétesse",
    "policière",
    "princesse",
    "brebis",
    "poule",
    "sculptrice",
    "madame",
    "fille",
    "jument",
    "belle-mère",
    "superwoman",
    "tigresse",
    "tante",
    "bonne",
    "serveuse",
    "administratrice web",
    "sorcière",
    "propriétaire",
    "hôtesse",
    "prêtresse",
    "abbesse",
    "comtesse",
    "nonne",
    "jeune fille",
    "femelle",
    "paonne",
    "hôtesse de l'air",
    "maîtresse",
    "veuve",
    "sultane",
    "vicomtesse",
    "belle-fille",
    "fiancée",
    "mariée",
    "vieille fille",
    "laitière",
    "bonne",
    "baronne",
    "vendeuse",
    "conductrice",
    "marraine",
    "petite amie",
    "bergère",
    "maman",
    "belle-fille",
    "filleule",
    "testatrice",
    "exécutrice",
    "chasseresse",
    "mairesse",
    "pair",
    "chanteuse",
    "tentatrice",
    "aventurière",
    "propriétaire",
    "géante",
    "marquise",
    "tsarine",
    "auteure",
    "enchanteresse",
    "couturière",
    "prophétesse",
]

# ---------------------------------------------------------------------------
# Domain / language partitions of the gendered vocab (for gender transfer)
# ---------------------------------------------------------------------------
# Index-aligned subsets of {eng,french}_{masc,fem}: the SAME index selects the
# same underlying concept across languages, so eng_*_prof and fr_*_prof are
# translation-aligned. Used by the cross-domain / cross-language gender transfer
# experiments — learn the gender axis on one domain/language, then measure
# held-out gender AUC on another WITHOUT refit (a transferable axis => the model
# captured "gender", not a domain/language artifact).
#
# Buckets chosen for semantic purity; generic-gender words (he/she, man/woman,
# sir/madam, proper nouns) are excluded. The French profession set drops indices
# whose French form has no gender distinction (propriétaire/propriétaire etc.).
_GENDER_PROF_IDX = [
    0, 7, 8, 11, 12, 14, 17, 23, 24, 25, 27, 30, 33, 35, 36, 37, 40, 48, 49,
    51, 53, 54, 55, 56, 57, 61, 62, 64, 65, 70, 71, 72, 73, 74, 77, 81, 82,
    83, 84, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96,
]  # 49 occupations / titles / agentive roles
_GENDER_KIN_IDX = [
    4, 9, 10, 13, 18, 19, 20, 21, 26, 34, 42, 44, 47, 63, 66, 67, 75, 76, 78,
    79, 80,
]  # 21 family / marital relations
_GENDER_ANIMAL_IDX = [2, 5, 6, 15, 28, 38, 39, 43, 46, 60]  # 10 animals (target-only)

eng_masc_prof = [eng_masc[i] for i in _GENDER_PROF_IDX]
eng_fem_prof = [eng_fem[i] for i in _GENDER_PROF_IDX]
eng_masc_kin = [eng_masc[i] for i in _GENDER_KIN_IDX]
eng_fem_kin = [eng_fem[i] for i in _GENDER_KIN_IDX]
eng_masc_animal = [eng_masc[i] for i in _GENDER_ANIMAL_IDX]
eng_fem_animal = [eng_fem[i] for i in _GENDER_ANIMAL_IDX]
fr_masc_prof = [french_masc[i] for i in _GENDER_PROF_IDX]
fr_fem_prof = [french_fem[i] for i in _GENDER_PROF_IDX]

CATEGORICAL_CODEBOOK = {
    "shape": [
        "spherical",
        "cuboidal",
        "conical",
        "circular",
        "squarish",
        "toroidal",
        "pyramidal",
        "cylindrical",
        "prismatic",
        "hemispherical",
    ],
    "color": [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "cyan",
        "teal",
        "lavender",
    ],
    "object": [
        "button",
        "shoe",
        "mug",
        "vase",
        "bead",
        "cushion",
        "toy",
        "statue",
        "drawing",
        "window",
    ],
}

# eng-french: household objects
# eng_french_obj = [
#     ("Friday", "Vendredi"),
#     ("Chair", "Chaise"),
#     ("Book", "Livre"),
#     ("Pen", "Stylo"),
#     ("Monday", "Lundi"),
#     ("flirt", "draguer"),
#     ("Computer", "Ordinateur"),
#     ("Kid", "gosse"),
#     ("Car", "Voiture"),
#     ("Bottle", "Bouteille"),
#     ("Cup", "Tasse"),
#     ("Bag", "Sac"),
#     ("Key", "Clé"),
#     ("Door", "Porte"),
#     ("Window", "Fenêtre"),
#     ("fish", "poisson"),
#     ("Pillow", "Oreiller"),
#     ("Blanket", "Couverture"),
#     ("Sunday", "Dimanche"),
#     ("Television", "Télévision"),
#     ("Clock", "Horloge"),
#     ("Watch", "Montre"),
#     ("Wallet", "Portefeuille"),
#     ("Shoe", "Chaussure"),
#     ("Dog", "Chien"),
#     ("Shirt", "Chemise"),
#     ("Pants", "Pantalon"),
#     ("Hat", "Chapeau"),
#     ("Umbrella", "Parapluie"),
#     ("Glasses", "Lunettes"),
#     ("Spoon", "Cuillère"),
#     ("Fork", "Fourchette"),
#     ("Knife", "Couteau"),
#     ("Plate", "Assiette"),
#     ("Bowl", "Bol"),
#     ("Toothbrush", "Brosse à dents"),
#     ("Toothpaste", "Dentifrice"),
#     ("Towel", "Serviette"),
#     ("Soap", "Savon"),
#     ("Shampoo", "Shampoing"),
#     ("Mirror", "Miroir"),
#     ("Comb", "Peigne"),
#     ("Brush", "Brosse"),
#     ("Candle", "Bougie"),
#     ("Saturday", "Samedi"),
#     ("Flower", "Fleur"),
#     ("Bicycle", "Vélo"),
#     ("Ball", "Balle"),
#     ("Toy", "Jouet"),
#     ("strong", "costaud"),
#     ("Microwave", "Micro-ondes"),
#     ("Oven", "Four"),
#     ("Refrigerator", "Réfrigérateur"),
#     ("Stove", "Cuisinière"),
#     ("Sink", "Évier"),
#     ("Toilet", "Toilettes"),
#     ("Bathtub", "Baignoire"),
#     ("Shower", "Douche"),
#     ("Desk", "Bureau"),
#     ("Notebook", "Cahier"),
#     ("girl", "fille"),
#     ("Stamp", "Timbre"),
#     ("Coin", "Pièce"),
#     ("Money", "Argent"),
#     ("toilets", "Chiottes"),
#     ("Remote", "Télécommande"),
#     ("Headphones", "Casque"),
#     ("Speaker", "Haut-parleur"),
#     ("strange", "Zarbi"),
#     ("Printer", "Imprimante"),
#     ("Mouse", "Souris"),
#     ("Keyboard", "Clavier"),
#     ("Monitor", "Écran"),
#     ("Laptop", "Ordinateur portable"),
#     ("man", "homme"),
#     ("Suitcase", "Valise"),
#     ("Map", "Carte"),
#     ("girlfriend", "meuf"),
#     ("cute", "mignonne"),
#     ("Painting", "Peinture"),
#     ("Thank you", "Merci beaucoup"),
#     ("Hammer", "Marteau"),
#     ("Screwdriver", "Tournevis"),
#     ("Wrench", "Clé"),
#     ("Drill", "Perceuse"),
#     ("Nail", "Clou"),
#     ("Screw", "Vis"),
#     ("Tape", "Ruban adhésif"),
#     ("Glue", "Colle"),
#     ("Scissors", "Ciseaux"),
#     ("Rope", "Corde"),
#     ("beautiful", "belle"),
#     ("Ladder", "Échelle"),
#     ("Bucket", "Seau"),
#     ("Mop", "Serpillière"),
#     ("Broom", "Balai"),
#     ("Trash Can", "Poubelle"),
#     ("lively", "charmante"),
#     ("Vacuum", "Aspirateur"),
#     ("Fan", "Ventilateur"),
# ]

# eng-german: household objects
# eng_german_obj = [
#     ("apple", "Apfel"),
#     ("book", "Buch"),
#     ("house", "Haus"),
#     ("cat", "Katze"),
#     ("dog", "Hund"),
#     ("car", "Auto"),
#     ("tree", "Baum"),
#     ("water", "Wasser"),
#     ("sky", "Himmel"),
#     ("sun", "Sonne"),
#     ("moon", "Mond"),
#     ("star", "Stern"),
#     ("school", "Schule"),
#     ("friend", "Freund"),
#     ("family", "Familie"),
#     ("bread", "Brot"),
#     ("butter", "Butter"),
#     ("cheese", "Käse"),
#     ("milk", "Milch"),
#     ("table", "Tisch"),
#     ("chair", "Stuhl"),
#     ("window", "Fenster"),
#     ("door", "Tür"),
#     ("pen", "Stift"),
#     ("paper", "Papier"),
#     ("flower", "Blume"),
#     ("road", "Straße"),
#     ("city", "Stadt"),
#     ("country", "Land"),
#     ("river", "Fluss"),
#     ("forest", "Wald"),
#     ("mountain", "Berg"),
#     ("fish", "Fisch"),
#     ("bird", "Vogel"),
#     ("horse", "Pferd"),
#     ("bicycle", "Fahrrad"),
#     ("train", "Zug"),
#     ("plane", "Flugzeug"),
#     ("boat", "Boot"),
#     ("ocean", "Ozean"),
#     ("rain", "Regen"),
#     ("snow", "Schnee"),
#     ("storm", "Sturm"),
#     ("cloud", "Wolke"),
#     ("earth", "Erde"),
#     ("fire", "Feuer"),
#     ("wind", "Wind"),
#     ("air", "Luft"),
#     ("music", "Musik"),
#     ("song", "Lied"),
#     ("dance", "Tanz"),
#     ("painting", "Gemälde"),
#     ("sculpture", "Skulptur"),
#     ("movie", "Film"),
#     ("bookstore", "Buchhandlung"),
#     ("university", "Universität"),
#     ("hospital", "Krankenhaus"),
#     ("library", "Bibliothek"),
#     ("museum", "Museum"),
#     ("restaurant", "Restaurant"),
#     ("kitchen", "Küche"),
#     ("bathroom", "Badezimmer"),
#     ("bedroom", "Schlafzimmer"),
#     ("garden", "Garten"),
#     ("computer", "Computer"),
#     ("phone", "Telefon"),
#     ("television", "Fernseher"),
#     ("radio", "Radio"),
#     ("lamp", "Lampe"),
#     ("clock", "Uhr"),
#     ("camera", "Kamera"),
#     ("mirror", "Spiegel"),
#     ("shirt", "Hemd"),
#     ("pants", "Hose"),
#     ("shoe", "Schuh"),
#     ("hat", "Hut"),
#     ("coat", "Mantel"),
#     ("glove", "Handschuh"),
#     ("scarf", "Schal"),
#     ("sock", "Socke"),
#     ("pillow", "Kissen"),
#     ("blanket", "Decke"),
#     ("knife", "Messer"),
#     ("fork", "Gabel"),
#     ("spoon", "Löffel"),
#     ("plate", "Teller"),
#     ("cup", "Tasse"),
#     ("bottle", "Flasche"),
#     ("glass", "Glas"),
#     ("ring", "Ring"),
#     ("necklace", "Halskette"),
#     ("bracelet", "Armband"),
#     ("soap", "Seife"),
#     ("towel", "Handtuch"),
#     ("toothbrush", "Zahnbürste"),
#     ("comb", "Kamm"),
#     ("shampoo", "Shampoo"),
#     ("key", "Schlüssel"),
#     ("doorbell", "Türklingel"),
#     ("envelope", "Umschlag"),
#     ("letter", "Brief"),
#     ("stamp", "Briefmarke"),
# ]

# gender: professions
# masc_fem = [
#     ["actor", "actress"],
#     ["batman", "batwoman"],
#     ["boar", "sow"],
#     ["boy", "girl"],
#     ["brother", "sister"],
#     ["buck", "doe"],
#     ["bull", "cow"],
#     ["businessman", "businesswoman"],
#     ["chairman", "chairwoman"],
#     ["dad", "mom"],
#     ["daddy", "mommy"],
#     ["duke", "duchess"],
#     ["emperor", "empress"],
#     ["father", "mother"],
#     ["fisherman", "fisherwoman"],
#     ["fox", "vixen"],
#     ["gentleman", "lady"],
#     ["god", "goddess"],
#     ["grandfather", "grandmother"],
#     ["grandpa", "grandma"],
#     ["grandson", "granddaughter"],
#     ["groom", "bride"],
#     ["he", "she"],
#     ["headmaster", "headmistress"],
#     ["heir", "heiress"],
#     ["hero", "heroine"],
#     ["husband", "wife"],
#     ["king", "queen"],
#     ["lion", "lioness"],
#     ["man", "woman"],
#     ["manager", "manageress"],
#     ["men", "women"],
#     ["mister", "miss"],
#     ["murderer", "murderess"],
#     ["nephew", "niece"],
#     ["poet", "poetess"],
#     ["policeman", "policewoman"],
#     ["prince", "princess"],
#     ["ram", "ewe"],
#     ["rooster", "hen"],
#     ["sculptor", "sculptress"],
#     ["sir", "madam"],
#     ["son", "daughter"],
#     ["stallion", "mare"],
#     ["stepfather", "stepmother"],
#     ["superman", "superwoman"],
#     ["tiger", "tigress"],
#     ["uncle", "aunt"],
#     ["valet", "maid"],
#     ["waiter", "waitress"],
#     ["webmaster", "webmistress"],
#     ["wizard", "witch"],
#     ["landlord", "landlady"],
#     ["host", "hostess"],
#     ["priest", "priestess"],
#     ["abbot", "abbess"],
#     ["count", "countess"],
#     ["monk", "nun"],
#     ["lad", "lass"],
#     ["male", "female"],
#     ["peacock", "peahen"],
#     ["steward", "stewardess"],
#     ["master", "mistress"],
#     ["widower", "widow"],
#     ["sultan", "sultana"],
#     ["earl", "countess"],
#     ["son-in-law", "daughter-in-law"],
#     ["fiancé", "fiancée"],
#     ["lord", "lady"],
#     ["bachelor", "spinster"],
#     ["milkman", "milkmaid"],
#     ["manservant", "maidservant"],
#     ["baron", "baroness"],
#     ["salesman", "saleswoman"],
#     ["conductor", "conductress"],
#     ["godfather", "godmother"],
#     ["boyfriend", "girlfriend"],
#     ["shepherd", "shepherdess"],
#     ["papa", "mama"],
#     ["stepson", "stepdaughter"],
#     ["godson", "goddaughter"],
#     ["testator", "testatrix"],
#     ["executor", "executrix"],
#     ["hunter", "huntress"],
#     ["mayor", "mayoress"],
#     ["peer", "peeress"],
#     ["songster", "songstress"],
#     ["tempter", "temptress"],
#     ["adventurer", "adventuress"],
#     ["proprietor", "proprietress"],
#     ["giant", "giantess"],
#     ["marquis", "marchioness"],
#     ["czar", "czarina"],
#     ["author", "authoress"],
#     ["enchanter", "enchantress"],
#     ["seamster", "seamstress"],
#     ["prophet", "prophetess"],
# ]

# eng -french + gender: professions
# binary_2 = [
#     [
#         "King",
#         "Reine",
#     ],  # English masculine word, French feminine translation (Queen)
#     ["Père", "Mother"],  # French masculine word, English feminine translation
#     [
#         "Brother",
#         "Sœur",
#     ],  # English masculine word, French feminine translation (Sister)
#     ["Tante", "Uncle"],  # French feminine word, English masculine translation
#     [
#         "Prince",
#         "Princesse",
#     ],  # English masculine word, French feminine translation (Princess)
#     [
#         "Actrice",
#         "Actor",
#     ],  # French feminine word, English masculine translation
#     [
#         "Serveuse",
#         "Waiter",
#     ],  # French feminine word, English masculine translation
#     ["Déesse", "God"],  # French feminine word, English masculine translation
#     [
#         "Wizard",
#         "Sorcière",
#     ],  # English masculine word, French feminine translation (Witch)
#     [
#         "Impératrice",
#         "Emperor",
#     ],  # French feminine word, English masculine translation
#     [
#         "Monk",
#         "Nonne",
#     ],  # English masculine word, French feminine translation (Nun)
#     ["Hôtesse", "Host"],  # French feminine word, English masculine translation
#     [
#         "Boy",
#         "Fille",
#     ],  # English masculine word, French feminine translation (Girl)
#     [
#         "Duchesse",
#         "Duke",
#     ],  # French feminine word, English masculine translation
#     [
#         "Widower",
#         "Veuve",
#     ],  # English masculine word, French feminine translation (Widow)
#     [
#         "Jument",
#         "Stallion",
#     ],  # French feminine word, English masculine translation
#     ["Vache", "Bull"],  # French feminine word, English masculine translation
#     [
#         "Rooster",
#         "Poule",
#     ],  # English masculine word, French feminine translation (Hen)
#     [
#         "Mariée",
#         "Bridegroom",
#     ],  # French feminine word, English masculine translation
#     [
#         "Dame",
#         "Gentleman",
#     ],  # French feminine word, English masculine translation
#     [
#         "Grandfather",
#         "Grand-mère",
#     ],  # English masculine word, French feminine translation (Grandmother)
#     ["Madame", "Sir"],  # French feminine word, English masculine translation
#     [
#         "Baron",
#         "Baronne",
#     ],  # English masculine word, French feminine translation (Baroness)
#     [
#         "Bergère",
#         "Shepherd",
#     ],  # French feminine word, English masculine translation
#     [
#         "Prêtresse",
#         "Priest",
#     ],  # French feminine word, English masculine translation
#     ["Lionne", "Lion"],  # French feminine word, English masculine translation
#     [
#         "Tiger",
#         "Tigresse",
#     ],  # English masculine word, French feminine translation (Tigress)
#     [
#         "Waiter",
#         "Infirmière",
#     ],  # English masculine word, French feminine translation (Nurse)
#     [
#         "Paonne",
#         "Peacock",
#     ],  # French feminine word, English masculine translation
#     ["Renarde", "Fox"],  # French feminine word, English masculine translation
#     [
#         "Marraine",
#         "Godfather",
#     ],  # French feminine word, English masculine translation
#     [
#         "Actor",
#         "Poétesse",
#     ],  # English masculine word, French feminine translation (Poetess)
#     [
#         "Duchesse",
#         "Wizard",
#     ],  # French feminine word, English masculine translation
#     [
#         "Queen",
#         "Roi",
#     ],  # English feminine word, French masculine translation (King)
#     ["Mère", "Father"],  # French feminine word, English masculine translation
#     [
#         "Sister",
#         "Frère",
#     ],  # English feminine word, French masculine translation (Brother)
#     ["Oncle", "Aunt"],  # French masculine word, English feminine translation
#     [
#         "Princess",
#         "Prince",
#     ],  # English feminine word, French masculine translation
#     [
#         "Acteur",
#         "Actress",
#     ],  # French masculine word, English feminine translation
#     [
#         "Waitress",
#         "Serveur",
#     ],  # English feminine word, French masculine translation (Waiter)
#     ["Neveu", "Niece"],  # French masculine word, English feminine translation
#     [
#         "Heroine",
#         "Héros",
#     ],  # English feminine word, French masculine translation (Hero)
#     [
#         "Goddess",
#         "Dieu",
#     ],  # English feminine word, French masculine translation (God)
#     [
#         "Witch",
#         "Sorcier",
#     ],  # English feminine word, French masculine translation (Wizard)
#     [
#         "Empereur",
#         "Empress",
#     ],  # French masculine word, English feminine translation
#     [
#         "Nun",
#         "Moine",
#     ],  # English feminine word, French masculine translation (Monk)
#     ["Hôte", "Hostess"],  # French masculine word, English feminine translation
#     [
#         "Girl",
#         "Garçon",
#     ],  # English feminine word, French masculine translation (Boy)
#     ["Duc", "Duchess"],  # French masculine word, English feminine translation
#     [
#         "Widow",
#         "Veuf",
#     ],  # English feminine word, French masculine translation (Widower)
#     [
#         "Mare",
#         "Étalon",
#     ],  # English feminine word, French masculine translation (Stallion)
#     ["Taureau", "Cow"],  # French masculine word, English feminine translation
#     [
#         "Hen",
#         "Coq",
#     ],  # English feminine word, French masculine translation (Rooster)
#     [
#         "Bride",
#         "Marié",
#     ],  # English feminine word, French masculine translation (Groom)
#     [
#         "Gentilhomme",
#         "Lady",
#     ],  # French masculine word, English feminine translation
#     [
#         "Grandmother",
#         "Grand-père",
#     ],  # English feminine word, French masculine translation (Grandfather)
#     [
#         "Monsieur",
#         "Madam",
#     ],  # French masculine word, English feminine translation
#     [
#         "Baron",
#         "Baroness",
#     ],  # French masculine word, English feminine translation
#     [
#         "Shepherdess",
#         "Berger",
#     ],  # English feminine word, French masculine translation (Shepherd)
#     [
#         "Prêtre",
#         "Priestess",
#     ],  # French masculine word, English feminine translation
#     ["Lioness", "Lion"],  # English feminine word, French masculine translation
#     [
#         "Tigress",
#         "Tigre",
#     ],  # English feminine word, French masculine translation (Tiger)
#     [
#         "Policier",
#         "Policewoman",
#     ],  # French masculine word, English feminine translation
#     [
#         "Peahen",
#         "Paon",
#     ],  # English feminine word, French masculine translation (Peacock)
#     ["Renard", "Vixen"],  # French masculine word, English feminine translation
#     [
#         "Parrain",
#         "Godmother",
#     ],  # French masculine word, English feminine translation
#     [
#         "Poète",
#         "Actress",
#     ],  # French masculine word, English feminine translation
#     [
#         "Duchess",
#         "Wizard",
#     ],  # English feminine word, French masculine translation
#     ["Emperor", "Impératrice"],
#     ["Fils", "Daughter"],
#     ["Mari", "Wife"],
#     ["Marié", "Bride"],
#     ["Dog", "Chienne"],
#     ["Stag", "Biche"],
#     ["Canard", "Duck"],
#     ["Garçon", "Girl"],
#     ["Homme", "Woman"],
#     # only_gender_english = [
#     ["actor", "actress"],
#     ["batman", "batwoman"],
#     ["boar", "sow"],
#     ["boy", "girl"],
#     ["brother", "sister"],
#     ["buck", "doe"],
#     ["bull", "cow"],
#     ["businessman", "businesswoman"],
#     ["chairman", "chairwoman"],
#     ["dad", "mom"],
#     ["daddy", "mommy"],
#     ["duke", "duchess"],
#     ["emperor", "empress"],
#     ["father", "mother"],
#     ["fisherman", "fisherwoman"],
#     ["fox", "vixen"],
#     ["gentleman", "lady"],
#     ["god", "goddess"],
#     ["grandfather", "grandmother"],
#     ["grandpa", "grandma"],
#     ["grandson", "granddaughter"],
#     ["groom", "bride"],
#     ["he", "she"],
#     ["headmaster", "headmistress"],
#     ["heir", "heiress"],
#     ["hero", "heroine"],
#     ["husband", "wife"],
#     ["king", "queen"],
#     ["lion", "lioness"],
#     ["man", "woman"],
#     ["manager", "manageress"],
#     ["men", "women"],
#     ["mister", "miss"],
#     ["murderer", "murderess"],
#     ["nephew", "niece"],
#     ["poet", "poetess"],
#     ["policeman", "policewoman"],
#     ["prince", "princess"],
#     ["ram", "ewe"],
#     ["rooster", "hen"],
#     ["sculptor", "sculptress"],
#     ["sir", "madam"],
#     ["son", "daughter"],
#     ["stallion", "mare"],
#     ["stepfather", "stepmother"],
#     ["superman", "superwoman"],
#     ["tiger", "tigress"],
#     ["uncle", "aunt"],
#     ["valet", "maid"],
#     ["waiter", "waitress"],
#     ["webmaster", "webmistress"],
#     # Additional unique pairs
#     ["wizard", "witch"],
#     ["landlord", "landlady"],
#     ["host", "hostess"],
#     ["priest", "priestess"],
#     ["abbot", "abbess"],
#     ["count", "countess"],
#     ["monk", "nun"],
#     ["lad", "lass"],
#     ["male", "female"],
#     ["peacock", "peahen"],
#     ["steward", "stewardess"],
#     ["master", "mistress"],
#     ["widower", "widow"],
#     ["sultan", "sultana"],
#     ["earl", "countess"],
#     ["son-in-law", "daughter-in-law"],
#     ["fiancé", "fiancée"],
#     ["lord", "lady"],
#     ["bachelor", "spinster"],
#     ["milkman", "milkmaid"],
#     ["manservant", "maidservant"],
#     ["baron", "baroness"],
#     ["salesman", "saleswoman"],
#     ["conductor", "conductress"],
#     ["godfather", "godmother"],
#     ["boyfriend", "girlfriend"],
#     ["shepherd", "shepherdess"],
#     ["papa", "mama"],
#     ["stepson", "stepdaughter"],
#     ["godson", "goddaughter"],
#     ["testator", "testatrix"],
#     ["executor", "executrix"],
#     ["hunter", "huntress"],
#     ["mayor", "mayoress"],
#     ["peer", "peeress"],
#     ["songster", "songstress"],
#     ["tempter", "temptress"],
#     ["adventurer", "adventuress"],
#     ["proprietor", "proprietress"],
#     ["giant", "giantess"],
#     ["marquis", "marchioness"],
#     ["czar", "czarina"],
#     ["author", "authoress"],
#     ["enchanter", "enchantress"],
#     ["seamster", "seamstress"],
#     ["prophet", "prophetess"],
#     # only_gender_french = [
#     ["roi", "reine"],  # king, queen
#     ["père", "mère"],  # father, mother
#     ["frère", "sœur"],  # brother, sister
#     ["oncle", "tante"],  # uncle, aunt
#     ["fils", "fille"],  # son, daughter
#     ["homme", "femme"],  # man, woman
#     ["garçon", "fille"],  # boy, girl
#     ["mari", "femme"],  # husband, wife
#     ["acteur", "actrice"],  # actor, actress
#     ["serveur", "serveuse"],  # waiter, waitress
#     ["hôte", "hôtesse"],  # host, hostess
#     ["dieu", "déesse"],  # god, goddess
#     ["héros", "héroïne"],  # hero, heroine
#     ["empereur", "impératrice"],  # emperor, empress
#     ["lion", "lionne"],  # lion, lioness
#     ["taureau", "vache"],  # bull, cow
#     ["cheval", "jument"],  # horse, mare
#     ["sorcier", "sorcière"],  # wizard, witch
#     ["moine", "nonne"],  # monk, nun
#     ["monsieur", "madame"],  # mister, madam
#     ["baron", "baronne"],  # baron, baroness
#     ["neveu", "nièce"],  # nephew, niece
#     ["grand-père", "grand-mère"],  # grandfather, grandmother
#     ["veuf", "veuve"],  # widower, widow
#     ["fiancé", "fiancée"],  # fiancé, fiancée
#     ["prince", "princesse"],  # prince, princess
#     ["duc", "duchesse"],  # duke, duchess
#     ["ami", "amie"],  # friend
#     ["voisin", "voisine"],  # neighbor
#     ["cousin", "cousine"],  # cousin
#     ["étudiant", "étudiante"],  # student
#     ["jumeau", "jumelle"],  # twin
#     ["boulanger", "boulangère"],  # baker
#     ["chanteur", "chanteuse"],  # singer
#     ["danseur", "danseuse"],  # dancer
#     ["vendeur", "vendeuse"],  # seller
#     ["docteur", "doctoresse"],  # doctor
#     ["ambassadeur", "ambassadrice"],  # ambassador
#     ["paysan", "paysanne"],  # peasant
#     ["tigre", "tigresse"],  # tiger, tigress
#     ["chien", "chienne"],  # dog
#     ["chat", "chatte"],  # cat
#     ["coq", "poule"],  # rooster, hen
#     ["bouc", "chèvre"],  # billy goat, goat
#     ["maître", "maîtresse"],  # master, mistress
#     ["lecteur", "lectrice"],  # reader
#     ["menteur", "menteuse"],  # liar
#     ["directeur", "directrice"],  # director
#     ["créateur", "créatrice"],  # creator
#     ["conducteur", "conductrice"],  # driver
#     ["cuisinier", "cuisinière"],  # cook
#     ["patron", "patronne"],  # boss
#     ["jardinier", "jardinière"],  # gardener
#     ["serveur", "serveuse"],  # waiter, waitress
#     ["boucher", "bouchère"],  # butcher
#     ["policier", "policière"],  # police officer
#     ["magicien", "magicienne"],  # magician
#     ["avocat", "avocate"],  # lawyer
#     ["professeur", "professeure"],  # teacher
#     ["ingénieur", "ingénieure"],  # engineer
#     ["docteur", "docteure"],  # doctor
#     ["président", "présidente"],  # president
#     ["artisan", "artisane"],  # artisan
#     ["comédien", "comédienne"],  # comedian
#     ["époux", "épouse"],  # husband, wife
#     ["général", "générale"],  # general
#     ["orateur", "oratrice"],  # speaker
#     ["pompier", "pompière"],  # firefighter
#     ["poète", "poétesse"],  # poet
#     ["chanteur", "chanteuse"],  # singer
#     ["agriculteur", "agricultrice"],  # farmer
#     ["vendeur", "vendeuse"],  # seller
#     ["musicien", "musicienne"],  # musician
#     ["boucher", "bouchère"],  # butcher
#     ["opticien", "opticienne"],  # optician
#     [
#         "sage-femme",
#         "maïeuticien",
#     ],  # midwife (note: male midwife is rare and called "maïeuticien")
#     ["hôte", "hôtesse"],  # host, hostess
#     ["aviateur", "aviatrice"],  # aviator
#     ["inspecteur", "inspectrice"],  # inspector
#     ["écrivain", "écrivaine"],  # writer
#     ["compositeur", "compositrice"],  # composer
#     ["éditeur", "éditrice"],  # editor
#     ["fabricant", "fabricante"],  # manufacturer
#     ["gérant", "gérante"],  # manager
#     ["chasseur", "chasseuse"],  # hunter
#     ["administrateur", "administratrice"],  # administrator
#     ["ambassadeur", "ambassadrice"],  # ambassador
#     ["bâtisseur", "bâtisseuse"],  # builder
#     ["commandant", "commandante"],  # commander
#     ["conseiller", "conseillère"],  # advisor
#     ["défenseur", "défenseure"],  # defender
#     ["entraîneur", "entraîneuse"],  # coach
#     ["gouverneur", "gouverneure"],  # governor
#     ["investisseur", "investisseuse"],  # investor
#     ["maçon", "maçonne"],  # mason
#     ["négociateur", "négociatrice"],  # negotiator
#     ["pêcheur", "pêcheuse"],  # fisherman
#     ["programmateur", "programmatrice"],  # programmer
#     ["représentant", "représentante"],  # representative
#     ["sage", "sage"],  # wise person (same word)
#     ["surveillant", "surveillante"],  # supervisor
#     ["traducteur", "traductrice"],  # translator
#     ["abbé", "abbesse"],
#     ["collègue", "consœur"],
#     ["cochon", "truie"],
#     ["loup", "louve"],
#     ["fondateur", "fondatrice"],
#     ["développeur", "développeuse"],
# ]
