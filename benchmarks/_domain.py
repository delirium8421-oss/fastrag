from typing import Dict, List

DOMAIN: Dict[str, str] = {
    "2wikimultihopqa": """Analyse the following passage and identify the people, creative works, and places mentioned in it. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 IMPORTANT: among other entities and relationships you find, make sure to extract as separate entities (to be connected with the main one) a person's
 role as a family member (such as 'son', 'uncle', 'wife', ...), their profession (such as 'director'), and the location
 where they live or work. Pay attention to the spelling of the names.""",  # noqa: E501
    "hotpotqa": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names."""
}

QUERIES: Dict[str, List[str]] = {
    "2wikimultihopqa": [
        "When did Prince Arthur's mother die?",
        "What is the place of birth of Elizabeth II's husband?",
        "Which film has the director died later, Interstellar or Harry Potter I?",
        "Where does the singer who wrote the song Blank Space work at?",
    ],
    "hotpotqa": [
        "Are Christopher Nolan and Sathish Kalathil both film directors?",
        "What language were books being translated into during the era of Haymo of Faversham?",
        "Who directed the film that was shot in or around Leland, North Carolina in 1986?",
        "Who wrote a song after attending a luau in the Koolauloa District on the island of Oahu in Honolulu County?"
    ]
}

ENTITY_TYPES: Dict[str, List[str]] = {
    "2wikimultihopqa": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
    ],
    "hotpotqa": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
}
