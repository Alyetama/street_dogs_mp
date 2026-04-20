#!/bin/bash

TARGET_DIR="grid_runs"
DELETE_ALL=false

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' not found in the current path."
    exit 1
fi

echo "Scanning for corrupted .gz files in '$TARGET_DIR'..."

while IFS= read -r -d $'\0' file; do
    if ! gzip -t "$file" 2>/dev/null; then

        if [ "$DELETE_ALL" = true ]; then
            rm "$file"
            echo "Deleted: $file"
        else
            echo "------------------------------------------------"
            echo "Corrupted file detected: $file"

            while true; do
                read -p "Delete this file? [y/n/A (yes to all)]: " choice </dev/tty

                case "$choice" in
                    [Yy]* )
                        rm "$file"
                        echo "Deleted: $file"
                        break
                        ;;
                    [Aa]* )
                        DELETE_ALL=true
                        rm "$file"
                        echo "Deleted: $file"
                        break
                        ;;
                    [Nn]* | "" )
                        echo "Skipped: $file"
                        break
                        ;;
                    * )
                        echo "Invalid input. Please answer y (yes), n (no), or a (yes to all)."
                        ;;
                esac
            done
        fi
    fi
done < <(find "$TARGET_DIR" -type f -name "*.gz" -print0)

echo "Scan complete."
