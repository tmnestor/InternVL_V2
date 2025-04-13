#!/usr/bin/env python3
"""
Migration script to update imports in files that depend on the old data_generators.

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
This script identifies files that import from the old data_generators module
and patches them to use the new ab initio implementation.
"""
import argparse
import os
import sys


def update_imports(file_path, dry_run=False):
    """
    Update imports in a file from data.data_generators to data.data_generators_new.
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, only show changes but don't apply them
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Map the major modules to their new versions
    replacements = {
        'data.data_generators.receipt_generator': 'data.data_generators_new.receipt_generator',
        'from data.data_generators.receipt_generator.create_receipt': 
            'from data.data_generators_new.receipt_generator.create_receipt',
        'create_tax_document': 'create_tax_document',
        'data.data_generators.create_multimodal_data': 
            'data.data_generators_new.create_multimodal_data'
    }
    
    new_content = content
    changes_made = False
    
    # Apply the replacements
    for old, new in replacements.items():
        if old in new_content:
            new_content = new_content.replace(old, new)
            changes_made = True
    
    # Handle function renames
    if 'create_receipt' in new_content:
        new_content = new_content.replace('create_receipt', 'create_receipt')
        changes_made = True
    
    # Add a comment about the migration
    if changes_made:
        migration_comment = (
            "# NOTE: This file has been updated to use the new ab initio implementation\n"
            "# of receipt and tax document generation. The old implementation in\n"
            "# data.data_generators has been replaced with data.data_generators_new.\n"
        )
        
        # Insert the comment after the imports
        import_section_end = new_content.find('\n\n', new_content.find('import'))
        if import_section_end > 0:
            # Split the string to avoid line length issues
            insert_point = import_section_end + 2
            new_content = new_content[:insert_point] + migration_comment + new_content[insert_point:]
        else:
            # Just add to the top if we can't find the import section end
            new_content = migration_comment + new_content
    
    if changes_made:
        if dry_run:
            print(f"Changes for {file_path}:")
            print("--- Original")
            print("+++ Modified")
            # Compare the original and modified content line by line
            original_lines = content.splitlines()
            modified_lines = new_content.splitlines()
            for _, (old_line, new_line) in enumerate(zip(original_lines, modified_lines, strict=False)):
                if old_line != new_line:
                    print(f"- {old_line}")
                    print(f"+ {new_line}")
        else:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Updated {file_path}")
    else:
        print(f"No changes needed for {file_path}")


def find_dependent_files(base_dir, pattern="data.data_generators"):
    """
    Find files that import from data.data_generators.
    
    Args:
        base_dir: Base directory to search
        pattern: Pattern to search for
        
    Returns:
        List of file paths
    """
    dependent_files = []
    
    for root, _, files in os.walk(base_dir):
        # Skip the data_generators directory itself
        if '/data/data_generators' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    if pattern in content:
                        dependent_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return dependent_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate data_generators imports")
    parser.add_argument("--base_dir", default=".", help="Base directory to search")
    parser.add_argument("--dry_run", action="store_true", help="Show changes without applying them")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Find files that import from data.data_generators
    dependent_files = find_dependent_files(args.base_dir)
    
    if not dependent_files:
        print("No files found that import from data.data_generators")
        sys.exit(0)
    
    print(f"Found {len(dependent_files)} files that import from data.data_generators:")
    for file in dependent_files:
        print(f"  {file}")
    
    # Update imports in dependent files
    if args.dry_run:
        print("\nDRY RUN - Showing changes but not applying them")
    
    for file in dependent_files:
        update_imports(file, args.dry_run)
    
    if args.dry_run:
        print("\nTo apply these changes, run without --dry_run")
    else:
        print("\nAll imports updated successfully")
        print("Make sure to verify the changes and test the code!")