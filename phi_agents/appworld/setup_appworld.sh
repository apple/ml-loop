#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

set -xe

echo $APPWORLD_ROOT
CUR_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash ${CUR_SCRIPT_DIR}/setup.sh

python -m virtualenv appworld-env
appworld-env/bin/pip install appworld
appworld-env/bin/appworld install
appworld-env/bin/appworld download data --root $APPWORLD_ROOT
