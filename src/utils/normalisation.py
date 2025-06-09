synonyms_map = {
    "graphene oxide": ["go", "GO", "graphene oxide"],
    "reduced graphene oxide": ["rgo", "RGO", "rGO", "reduced graphene oxide"],
    "CVD": ["chemical vapor deposition", "CVD"]
}

def unify_fabrication_label(label: str) -> str:
    label_lower = label.lower()
    for canonical, synonyms in synonyms_map.items():
        if label_lower in [s.lower() for s in synonyms]:
            return canonical
    return label  # If no match, return original