#!/bin/bash

# Build main
make build

# Define ANSI color codes
green="\033[32m"
red="\033[31m"
reset="\033[0m"

has_failed=false

# Traverse the tests/ folder and look for subfolders
for testfolder in tests/test_*; do

  # Extract the test name from the folder name
  testname=$(basename "$testfolder" | sed 's/^test_//')
  
  # Define the input and expected output filenames
  inputfile="${testfolder}/test.pcl"
  expectedfile="${testfolder}/expected.txt"
  
  # Run the test and capture the output
  output=$(./pycimen "$inputfile")
  
  # Compare the output to the expected output
  expected=$(cat "$expectedfile")
  
  if [ "$output" = "$expected" ]; then
    printf "${green}Test ${testname}: PASSED${reset}\n"
  else
    printf "${red}Test ${testname}: FAILED${reset}\n"
    printf "${red}Expected output:${reset}\n"
    printf "$expected\n"
    printf "${red}Actual output:${reset}\n"
    printf "$output\n"
    has_failed=true
  fi
done

if [ "$has_failed" = false ]; then
  printf "${green}All tests passed!${reset}\n"
  exit 0
else
  printf "${red}Test failure.${reset}\n"
  exit 1
fi
