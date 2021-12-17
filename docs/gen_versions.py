import json
from pathlib import Path


def main():
    versions = [
        str(x) for x in Path('.').glob('*') 
        if x.is_dir() and str(x) not in ('_build', 'src')
    ]
    with open('versions.json', 'w') as f:
        json.dump(versions, f)


if __name__ == '__main__':
    main()
