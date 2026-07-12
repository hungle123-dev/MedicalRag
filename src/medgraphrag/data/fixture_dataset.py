"""Synthetic MIRAGE-shaped demo data (NOT real MIRAGE).

Questions paraphrase the corpus (BM25 must do real work); corpus has
distractor docs mentioning wrong options; each question has gold_terms so
retrieval is scored independently of answer accuracy. Triples form a small KG
for the graph arm; some questions need a 2-hop path (multi-hop) so PPR has a
chance to matter.
"""
from medgraphrag.core.types import Question, GraphTriple

QUESTIONS = [
    Question("q1", "Which agent is preferred for streptococcal pharyngitis?",
             {"A": "penicillin", "B": "insulin", "C": "warfarin", "D": "aspirin"}, "A",
             gold_terms=("penicillin",)),
    Question("q2", "Which pancreatic hormone lowers serum glucose?",
             {"A": "glucagon", "B": "insulin", "C": "cortisol", "D": "thyroxine"}, "B",
             gold_terms=("insulin",)),
    Question("q3", "Which oral agent antagonizes vitamin K to prevent clotting?",
             {"A": "aspirin", "B": "heparin", "C": "warfarin", "D": "penicillin"}, "C",
             gold_terms=("warfarin",)),
    Question("q4", "Which drug blocks cyclooxygenase to impair platelets?",
             {"A": "aspirin", "B": "insulin", "C": "glucagon", "D": "heparin"}, "A",
             gold_terms=("aspirin",)),
    Question("q5", "Which pancreatic hormone raises serum glucose while fasting?",
             {"A": "insulin", "B": "glucagon", "C": "warfarin", "D": "penicillin"}, "B",
             gold_terms=("glucagon",)),
    # 2-hop: drug class -> member -> mechanism. Needs graph traversal.
    Question("q6", "A patient needs a fast injectable anticoagulant; which drug?",
             {"A": "warfarin", "B": "aspirin", "C": "heparin", "D": "insulin"}, "C",
             gold_terms=("heparin",)),
]

CORPUS = {
    "d1": "For sore throat from group A streptococcus, penicillin is the recommended "
          "treatment; aspirin only relieves pain.",
    "d2": "After meals, insulin from the pancreas drives glucose into cells and lowers "
          "blood sugar, opposing glucagon.",
    "d3": "Warfarin blocks vitamin K dependent clotting factors, whereas heparin acts "
          "through a faster pathway.",
    "d4": "Aspirin irreversibly acetylates cyclooxygenase, cutting thromboxane and "
          "platelet activation; penicillin has no such effect.",
    "d5": "While fasting the pancreas releases glucagon, which raises blood sugar, "
          "unlike insulin.",
    "d6": "Heparin is an injectable anticoagulant with rapid onset, used when speed "
          "matters, in contrast to oral warfarin.",
}

TRIPLES = [
    GraphTriple("streptococcal pharyngitis", "treated_with", "penicillin"),
    GraphTriple("serum glucose", "lowered_by", "insulin"),
    GraphTriple("vitamin K", "antagonized_by", "warfarin"),
    GraphTriple("cyclooxygenase", "blocked_by", "aspirin"),
    GraphTriple("serum glucose", "raised_by", "glucagon"),
    # 2-hop chain for q6: injectable anticoagulant -> heparin -> rapid onset
    GraphTriple("anticoagulant", "injectable_member", "heparin"),
    GraphTriple("heparin", "has_property", "rapid onset"),
]
