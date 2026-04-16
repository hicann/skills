#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ARCHIVE_PATH="${REPO_ROOT}/agent/assets/ops-easyasc-dsl-runtime.tar.gz"
EXAMPLE_ARCHIVE_PATH="${REPO_ROOT}/agent/assets/ops-easyasc-dsl-example.tar.gz"

if [[ ! -f "${RUNTIME_ARCHIVE_PATH}" ]]; then
    echo "archive not found: ${RUNTIME_ARCHIVE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${EXAMPLE_ARCHIVE_PATH}" ]]; then
    echo "archive not found: ${EXAMPLE_ARCHIVE_PATH}" >&2
    exit 1
fi

missing_runtime=0
for path in easyasc doc doc_cn; do
    if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
        missing_runtime=1
        break
    fi
done

missing_example=0
if [[ ! -e "${REPO_ROOT}/agent/example" ]]; then
    missing_example=1
fi

if [[ "${missing_runtime}" -eq 0 && "${missing_example}" -eq 0 ]]; then
    echo "easyasc/, doc/, doc_cn/, and agent/example/ are already present."
    exit 0
fi

if [[ "${missing_runtime}" -eq 1 ]]; then
    tar -xzf "${RUNTIME_ARCHIVE_PATH}" -C "${REPO_ROOT}" --skip-old-files
    echo "restored easyasc/, doc/, and doc_cn/ from ${RUNTIME_ARCHIVE_PATH}"
fi

if [[ "${missing_example}" -eq 1 ]]; then
    tar -xzf "${EXAMPLE_ARCHIVE_PATH}" -C "${REPO_ROOT}" --skip-old-files
    echo "restored agent/example/ from ${EXAMPLE_ARCHIVE_PATH}"
fi
