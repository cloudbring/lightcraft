#!/bin/sh
case "$*" in
  "run -q -p lightcraft-cli -- commands --json")
    printf '[{"id":"edit.undo"},{"id":"library.import"}]\n' ;;
  "run -q -p lightcraft-cli -- controls --json")
    printf '[{"id":"light.exposure"}]\n' ;;
  *) printf 'unexpected cargo invocation: %s\n' "$*" >&2; exit 1 ;;
esac
