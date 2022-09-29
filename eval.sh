#! /bin/bash

tmpfile=$(mktemp /tmp/qcfg_eval.XXXXXX)

awk -F'\t' '{print $2}' "$1" > "$tmpfile"
nlg-eval --hypothesis "$tmpfile" --references "$2"

rm "$tmpfile"
