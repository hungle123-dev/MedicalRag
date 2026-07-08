"""Synthetic MIRAGE-shaped demo dataset (NOT real MIRAGE data).

Lives in src so both the runner and tests import from one place. A real
MIRAGE loader is added alongside this later under the same load_dataset seam.
Each gold answer's option-text appears in exactly one corpus doc and one triple.
"""
from medgraphrag.core.types import Question

QUESTIONS = [
    Question("q1", "First-line antibiotic for strep throat?",
             {"A": "penicillin", "B": "insulin", "C": "warfarin", "D": "aspirin"}, "A"),
    Question("q2", "Hormone that lowers blood glucose?",
             {"A": "glucagon", "B": "insulin", "C": "cortisol", "D": "thyroxine"}, "B"),
    Question("q3", "Anticoagulant that inhibits vitamin K?",
             {"A": "aspirin", "B": "heparin", "C": "warfarin", "D": "penicillin"}, "C"),
    Question("q4", "Drug that irreversibly inhibits COX to reduce platelet aggregation?",
             {"A": "aspirin", "B": "insulin", "C": "glucagon", "D": "heparin"}, "A"),
    Question("q5", "Hormone that raises blood glucose?",
             {"A": "insulin", "B": "glucagon", "C": "warfarin", "D": "penicillin"}, "B"),
    Question("q6", "Parenteral anticoagulant potentiating antithrombin III?",
             {"A": "warfarin", "B": "aspirin", "C": "heparin", "D": "insulin"}, "C"),
]

CORPUS = {
    "d1": "Penicillin is the first-line antibiotic for streptococcal pharyngitis.",
    "d2": "Insulin is the hormone that lowers blood glucose after meals.",
    "d3": "Warfarin is an oral anticoagulant that inhibits vitamin K epoxide reductase.",
    "d4": "Aspirin irreversibly inhibits COX and reduces platelet aggregation.",
    "d5": "Glucagon is the hormone that raises blood glucose during fasting.",
    "d6": "Heparin is a parenteral anticoagulant that potentiates antithrombin III.",
}

TRIPLES = [
    ("strep throat", "first_line_treatment", "penicillin"),
    ("blood glucose", "lowered_by", "insulin"),
    ("vitamin K", "inhibited_by", "warfarin"),
    ("platelet aggregation", "reduced_by", "aspirin"),
    ("blood glucose", "raised_by", "glucagon"),
    ("antithrombin III", "potentiated_by", "heparin"),
]
