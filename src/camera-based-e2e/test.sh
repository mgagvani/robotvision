#!/bin/bash

# Configuration
PARTITIONS=("v100" "a10" "a30")
ACCOUNT="csso"
GRES="gpu:1"
MEM="128G"
CPUS="16"
TIME="8:00:00"

echo "Checking estimated wait times for 8h job..."
echo "--------------------------------------------"

for PART in "${PARTITIONS[@]}"; do
    NOW_TS=$(date +%s)

    # Run sbatch test-only
    OUTPUT=$(sbatch --test-only \
        -A $ACCOUNT \
        --gres=$GRES \
        --mem=$MEM \
        --cpus-per-task=$CPUS \
        --partition=$PART \
        -t $TIME \
        --wrap="hostname" 2>&1)

    if [[ "$OUTPUT" == *"allocation available"* ]] || [[ "$OUTPUT" == *"begins now"* ]]; then
        echo "${PART}: 0:00:00 delay (Immediate)"
    else
        # 1. Capture the text after "start at"
        EST_TIME_STRING=$(echo "$OUTPUT" | grep -oP 'start at \K.*')

        # 2. FIX: Isolate ONLY the timestamp (take the first "word" of the string)
        # This removes "using 16 processors on nodes..."
        CLEAN_TIMESTAMP=$(echo "$EST_TIME_STRING" | awk '{print $1}')

        if [[ -z "$CLEAN_TIMESTAMP" ]]; then
            echo "${PART}: [Could not parse SLURM output] $OUTPUT"
        else
            # 3. Convert clean timestamp to Epoch seconds
            EST_TS=$(date -d "$CLEAN_TIMESTAMP" +%s 2>/dev/null)

            if [[ -z "$EST_TS" ]]; then
                 echo "${PART}: [Date parse error] Raw: $CLEAN_TIMESTAMP"
            else
                DIFF=$((EST_TS - NOW_TS))
                if [ $DIFF -lt 0 ]; then DIFF=0; fi

                # Format output
                FORMATTED_DELAY=$(date -u -d @"$DIFF" +'%H:%M:%S')
                DAYS=$((DIFF / 86400))

                if [ $DAYS -gt 0 ]; then
                    FORMATTED_DELAY="${DAYS}d-${FORMATTED_DELAY}"
                fi

                echo "${PART}: ${FORMATTED_DELAY} delay (Est: $CLEAN_TIMESTAMP)"
            fi
        fi
    fi
done