#!/usr/bin/env bash
# Redireciona para o runner principal na raiz do projeto
here="$(dirname "$(readlink -f "$0")")"
exec bash "$here/../run_visual.sh" "$@"
