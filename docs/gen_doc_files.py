import os


def make_docfile(name):
    modname = name.replace('/', '.')
    level = modname.count('.')
    print(modname, level)
    with open(f'src/apidocs/{modname}.md', 'w') as f:
        f.write(f'##{level * "#"} `{modname}`\n'
                '\n'
                '```{eval-rst}\n'
                f'.. automodule:: {modname}\n'
                '    :members:\n'
                '```\n')


def main():
    for x in os.walk('../padl'):
        for f in x[2]:
            folder = x[0][3:]
            if f.endswith('.py'):
                if f == '__init__.py':
                    make_docfile(f'{folder}')
                else:
                    make_docfile(f'{folder}/{f.split(".py")[0]}')


if __name__ == '__main__':
    main()
