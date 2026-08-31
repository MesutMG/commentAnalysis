#!/usr/bin/env bash

BASE_URL="http://localhost:1967"
CSV_FILE="data/example_comments.csv"

echo "=== 1. Testing Single Inappropriate / Negative Comment ==="
curl -s -X POST "$BASE_URL/analyzeComment/" \
     -H "Content-Type: application/json" \
     -d '{
       "comment": "Lavabolar berbat hijyen sıfır...",
       "star": 5.0
     }' | jq . || cat

echo -e "\n\n=== 2. Testing Single Positive Comment ==="
curl -s -X POST "$BASE_URL/analyzeComment/" \
     -H "Content-Type: application/json" \
     -d '{
       "comment": "Uçuş konforlu ve fiyatlar çok uygundu teşekkürler.",
       "star": 1.0
     }' | jq . || cat

echo -e "\n\n=== 3. Testing Whole CSV Upload ==="
if [ -f "$CSV_FILE" ]; then
    curl -s -X POST "$BASE_URL/analyzeWholeCsv/" \
         -F "file=@$CSV_FILE" | jq . || cat
else
    echo "Warning: '$CSV_FILE' not found, skipping CSV upload test."
fi

echo -e "\n"