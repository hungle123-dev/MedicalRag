"""Synthetic MIRAGE-shaped demo dataset (NOT real MIRAGE data).

Design goals (per adversarial review):
- Questions PARAPHRASE the corpus, they do not copy its wording, so BM25 must
  do real lexical work rather than exact-substring matching.
- The corpus contains DISTRACTOR docs that mention wrong options, so a naive
  retriever can be fooled.
- Each question carries gold_terms identifying the correct evidence, so
  retrieval quality is scored independently of answer accuracy.
Still synthetic — a real MIRAGE loader replaces this behind load_dataset.
"""
from medgraphrag.core.types import Question

QUESTIONS = [
    Question("q1", "Which agent is preferred to treat streptococcal pharyngitis?",
             {"A": "penicillin", "B": "insulin", "C": "warfarin", "D": "aspirin"}, "A",
             gold_terms=("penicillin",)),
    Question("q2", "Which pancreatic hormone reduces serum glucose?",
             {"A": "glucagon", "B": "insulin", "C": "cortisol", "D": "thyroxine"}, "B",
             gold_terms=("insulin",)),
    Question("q3", "Which oral agent antagonizes vitamin K to prevent clots?",
             {"A": "aspirin", "B": "heparin", "C": "warfarin", "D": "penicillin"}, "C",
             gold_terms=("warfarin",)),
    Question("q4", "Which drug blocks cyclooxygenase to impair platelet function?",
             {"A": "aspirin", "B": "insulin", "C": "glucagon", "D": "heparin"}, "A",
             gold_terms=("aspirin",)),
    Question("q5", "Which pancreatic hormone elevates serum glucose during fasting?",
             {"A": "insulin", "B": "glucagon", "C": "warfarin", "D": "penicillin"}, "B",
             gold_terms=("glucagon",)),
    Question("q6", "Which injectable agent boosts antithrombin activity?",
             {"A": "warfarin", "B": "aspirin", "C": "heparin", "D": "insulin"}, "C",
             gold_terms=("heparin",)),
]

# Docs paraphrase questions; several mention MULTIPLE drugs (distractors).
CORPUS = {
    "d1": "For sore throat caused by group A streptococcus, penicillin remains "
          "the recommended treatment; aspirin only relieves the pain.",
    "d2": "After a carbohydrate meal, insulin secreted by the pancreas drives "
          "glucose into cells and lowers blood sugar, opposing glucagon.",
    "d3": "Warfarin blocks the vitamin K dependent clotting factors, whereas "
          "heparin acts through a different, faster pathway.",
    "d4": "Aspirin acetylates cyclooxygenase irreversibly, reducing thromboxane "
          "and thereby platelet activation; penicillin has no such effect.",
    "d5": "During fasting the pancreas releases glucagon, which raises blood "
          "sugar by promoting hepatic glycogen breakdown, unlike insulin.",
    "d6": "Heparin, an injectable anticoagulant, accelerates antithrombin and is "
          "used when rapid effect is needed, in contrast to oral warfarin.",
}

# Triples carry a RELATION that matches the question's intent; several share a
# head ("blood glucose") so relation must disambiguate lowered vs raised.
TRIPLES = [
    ("streptococcal pharyngitis", "treated_with", "penicillin"),
    ("serum glucose", "reduced_by", "insulin"),
    ("vitamin K", "antagonized_by", "warfarin"),
    ("cyclooxygenase", "blocked_by", "aspirin"),
    ("serum glucose", "elevated_by", "glucagon"),
    ("antithrombin", "activated_by", "heparin"),
]
