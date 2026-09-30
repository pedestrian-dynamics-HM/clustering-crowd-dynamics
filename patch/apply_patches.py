# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import click
import os
import subprocess

import datafold


def do_link(file_path, force=False, local_path=None, packagent=None):
    file_path = os.path.abspath(os.path.join(packagent.__file__, f'../{file_path}'))
    # Infer local_path automatically.
    if local_path is None:
        local_path = f'../{file_path}'
    local_home = os.path.abspath(os.path.join(__file__, f'../{local_path}'))
    # If installed package dir does not exist, continue either way. We'll
    # remove it/create a link from there anyways.
    if not os.path.isfile(file_path):
        print(f'{file_path} does not exist. Continuing to link.')
    # Make sure the path we are linking to does exist.
    assert os.path.exists(local_home), local_home
    # Confirm with user.
    if not force and not click.confirm(
        f'This will replace:\n  {file_path}\nwith a symlink to:\n  {local_home}', default=True
    ):
        return

    # create the links
    sudo = []
    if not os.access(os.path.dirname(file_path), os.W_OK):
        print(f"You don't have write permission to {file_path}, using sudo:")
        sudo = ['sudo']
    print(f'Creating symbolic link from \n {local_home} to \n {file_path}')
    subprocess.check_call(sudo + ['rm', '-rf', file_path])
    subprocess.check_call(sudo + ['ln', '-s', local_home, file_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Setup dev.')
    parser.add_argument('--yes', '-y', action='store_true', help='datafold patches.')
    args = parser.parse_args()

    # update collections for storing models properly
    # issue with custom TSCDataFrame and is_validate property
    do_link(
        'pcfold/timeseries/collection.py',
        force=args.yes,
        local_path='./datafold/pcfold/timeseries/collection.py',
        packagent=datafold,
    )

    do_link('dynfold/dmd.py', force=args.yes, local_path='./datafold/dynfold/dmd.py', packagent=datafold)
    do_link('dynfold/base.py', force=args.yes, local_path='./datafold/dynfold/base.py', packagent=datafold)

    do_link('dynfold/__init__.py', force=args.yes, local_path='./datafold/dynfold/__init__.py', packagent=datafold)

    print('finish soft link')
