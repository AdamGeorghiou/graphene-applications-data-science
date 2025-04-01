import os
import sys
import traceback

# Add the project root to Python's path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Try to import each module one by one to see where it fails
modules_to_test = [
    "src.processors.nlp.base_processor",
    "src.processors.nlp.entity_extractor",
    "src.processors.nlp.application_classifier",
    "src.processors.nlp.relation_extractor",
    "src.processors.nlp.trl_assessor",
    "src.processors.nlp.document_processor"
]

for module_name in modules_to_test:
    print(f"Trying to import {module_name}...")
    try:
        module = __import__(module_name, fromlist=["*"])
        print(f"✅ Successfully imported {module_name}")
        print(f"   Defined names: {dir(module)}")
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        print(traceback.format_exc())
    print("-" * 50)