#! /bin/bash

# cli

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
  -f | --full)
    FULL=YES
    shift # past argument
    ;;
  *) # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift              # past argument
    ;;
  esac
done

# path
folder=${PWD##*/}
declare -A targets=(["group1"]="group1:~/${folder}" ["ai1"]="ai:~/project/${folder}")

if [ ${#POSITIONAL[@]} -eq 0 ]; then
  echo "No target provided. Sync all."
  POSITIONAL=("${!targets[@]}")
fi

for target in "${POSITIONAL[@]}"; do
  echo "==============="
  echo "syncing ${target}"
  if [ -z "${targets[$target]}" ]; then
    echo "Unrecognized ${target}. Skipping."
    continue
  fi
  rsync -avhHLP "$PWD"/ "${targets[$target]}" --exclude={__pycache__,logs,.git,cache,.idea,/data,.vscode}
  if [ "${FULL}" = YES ]; then
    rsync -avhHLP "$PWD"/data "${targets[$target]}" --exclude={.idea,.git,__pycache__,cache}
  fi
done
