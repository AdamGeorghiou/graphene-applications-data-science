#!/bin/bash
# Setup script for Graphene Applications Collection system

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p "$PROJECT_ROOT/src/data_collection"
mkdir -p "$PROJECT_ROOT/data/raw/merged"
mkdir -p "$PROJECT_ROOT/data/raw/arxiv"
mkdir -p "$PROJECT_ROOT/data/raw/scopus"
mkdir -p "$PROJECT_ROOT/data/raw/patents"
mkdir -p "$PROJECT_ROOT/data/raw/semantic_scholar"
mkdir -p "$PROJECT_ROOT/data/raw/ieee"
mkdir -p "$PROJECT_ROOT/logs"

# Copy files to appropriate locations
echo "Setting up collection scripts..."

# Create data_collection directory if it doesn't exist
if [ ! -d "$PROJECT_ROOT/src/data_collection" ]; then
    mkdir -p "$PROJECT_ROOT/src/data_collection"
fi

# Copy unified collector to data_collection directory
cp "$PROJECT_ROOT/src/unified_collector.py" "$PROJECT_ROOT/src/data_collection/" 2>/dev/null || :
cp "$PROJECT_ROOT/src/domain_specific_collector.py" "$PROJECT_ROOT/src/data_collection/" 2>/dev/null || :
cp "$PROJECT_ROOT/src/batch_collection.py" "$PROJECT_ROOT/src/data_collection/" 2>/dev/null || :

# Copy collection config to config directory
if [ ! -d "$PROJECT_ROOT/src/config" ]; then
    mkdir -p "$PROJECT_ROOT/src/config"
fi
cp "$PROJECT_ROOT/src/collection_config.py" "$PROJECT_ROOT/src/config/" 2>/dev/null || :

# Add __init__.py files
touch "$PROJECT_ROOT/src/data_collection/__init__.py"
touch "$PROJECT_ROOT/src/config/__init__.py"

# Make scripts executable
chmod +x "$PROJECT_ROOT/src/data_collection/unified_collector.py" 2>/dev/null || :
chmod +x "$PROJECT_ROOT/src/data_collection/domain_specific_collector.py" 2>/dev/null || :
chmod +x "$PROJECT_ROOT/src/data_collection/batch_collection.py" 2>/dev/null || :

echo "Setup complete!"
echo
echo "To run a collection, use one of the following commands:"
echo
echo "  # Run a general collection:"
echo "  python src/data_collection/batch_collection.py --size medium --mode general"
echo
echo "  # Run a domain-specific collection:"
echo "  python src/data_collection/batch_collection.py --size medium --mode domain --domains energy electronics materials"
echo
echo "  # Run a balanced collection:"
echo "  python src/data_collection/batch_collection.py --size medium --mode balanced"
echo