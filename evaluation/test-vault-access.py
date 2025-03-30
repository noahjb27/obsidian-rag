def test_vault_access():
    from pathlib import Path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    
    vault_path = Path(config.DEFAULT_VAULT_PATH).expanduser().resolve()
    print(f"Testing access to Obsidian vault at: {vault_path}")
    
    if vault_path.exists():
        print(f"✓ Vault directory exists")
        md_files = list(vault_path.glob('**/*.md'))
        print(f"✓ Found {len(md_files)} markdown files")
        if md_files:
            print(f"✓ Sample file: {md_files[0]}")
    else:
        print(f"✗ Vault directory not found at {vault_path}")

if __name__ == "__main__":
    test_vault_access()