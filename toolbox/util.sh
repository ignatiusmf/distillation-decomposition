#!/bin/bash
# Cleans up stale experiment directories on CHPC.
# A directory is deleted if: it has no subdirectories, no Accuracy.png (i.e. never finished
# a full epoch), and no active PBS job pointing to it. Run this manually when needed.

# Build a set of experiment paths that currently have active PBS jobs.
declare -A jobnames
for job in $(qstat -u iferreira | awk 'NR>5 {print $1}'); do
    name=$(qstat -f "$job" | awk -F'= ' '/Job_Name =/ {print $2}' | tr -d ' ')
    if [[ -n "$name" ]]; then
        echo "Currently queued: experiments/$name"
        jobnames["experiments/$name"]=1
    fi
done

# Repeatedly scan experiments/ and delete empty, result-free, idle dirs.
# Loop repeats until a full pass finds nothing to delete (handles nested empties).
deleted_any=true

while $deleted_any; do
    deleted_any=false
    while read -r dir; do
        if [[ -z "$(find "$dir" -mindepth 1 -type d)" && \
              ! -f "$dir/Accuracy.png" && \
              -z "${jobnames[$dir]}" ]]; then
            echo "Deleting $dir"
            rm -rf "$dir"
            deleted_any=true
        fi
    done < <(find experiments/* -type d)
done