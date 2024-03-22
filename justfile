# List recipes
default:
    just --list

# Generate lock file for cpu|cuda|rocml
lock target:
    pdm lock -G visualize -G {{target}} -L pdm.{{target}}.lock

# Generate all lock files
lock-all:
    just lock cpu
    just lock cuda
    # just lock rocm

# install dev deps between cpu|cuda|rocml
install-dev target:
    pdm install -G visualize -L pdm.{{target}}.lock
