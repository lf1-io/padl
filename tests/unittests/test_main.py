import os
import subprocess


def test_aleph_cli_help():
    result = subprocess.run(['python', '-m', 'lf', '--help'], capture_output=True)
    assert 'Show this message and exit.' in result.stdout.decode()

    result = subprocess.run(['python', '-m', 'lf', 'check', '--help'], capture_output=True)
    assert 'Check the model by generating inputs' in result.stdout.decode()

    result = subprocess.run(['python', '-m', 'lf', 'check', '-m',
                             'tests/material/string_to_tensor.tabc'],
                            capture_output=True)
    assert 'output: tensor([' in result.stdout.decode()

    result = subprocess.run(['python', '-m', 'lf', 'compute-outputs',
                             '-m', 'tests/material/string_to_tensor.tabc',
                             '-i', 'tests/material/input.txt',
                             '-o', 'tests/material/out',
                             '-v', '',
                             '-p', 'txt',
                             '-s', 'numpy'],
                            capture_output=True)
    os.remove('tests/material/out.npy')
    assert 'found saver/ loader version' in result.stdout.decode()
