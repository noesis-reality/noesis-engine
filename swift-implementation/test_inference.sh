#!/bin/bash

echo "🧪 Testing Noesis Engine Inference"
echo "=================================="
echo ""

# Test 1: Simple math
echo "Test 1: Simple math (1+1=)"
echo -n "Result: "
.build/release/noesis-generate "1+1=" --max-tokens 1 --temperature 0 2>&1 | tail -1
echo ""

# Test 2: Capital question  
echo "Test 2: Capital of France"
echo -n "Result: "
timeout 30 .build/release/noesis-generate "The capital of France is" --max-tokens 3 --temperature 0 2>&1 | tail -1
echo ""

# Test 3: With stats
echo "Test 3: Generation with stats"
.build/release/noesis-generate "Hello" --max-tokens 5 --temperature 0.7 --stats 2>&1 | grep -E "(Tokens generated|Speed)" || echo "Stats output not found"
echo ""

echo "✅ Inference tests complete!"