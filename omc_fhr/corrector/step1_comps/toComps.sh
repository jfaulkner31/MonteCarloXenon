#!/bin/bash

#########################################
######## Made by Jonathon Faulkner ######
######## SCALE version 6.3.1 ############
######## or 6.2.4 #######################
######## 11-27-2024 #####################
#########################################
# Grabs a set of nuclides, as defined by a dictionary, and dumps them into the STDCMP format

# Examples:
# material id -> 101
# temperature -> 900.0
# f71 file we want to grab data from -> PREDICTOR_EOS_step0_mat101.f71
# use a dictionary to only pull certain nuclides from the dict -> addnuxDicts/addnux2Dict.dict
# Scale version number -> 624, 631, or 632
# then do:dd
# bash f71_to_comp.sh 101 900.0 PREDICTOR_EOS_step0_mat101.f71 addnuxDicts/addnux2Dict.dict 624

# see OBIWAN manual of scale for details

# to call obiwan script example (use 624 or 631 or 632 for scale name)
# bash f71_to_comp.sh 101 900.0 myf71file.f71 myAddnuxDict.dict myOutputFilename 624

# ascii art
# printf '             NOW RUNNING F71 CONVERSIONS \n'
# printf '  /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ \n'
# printf '  \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ \n'
# printf '  /\/\             /\    /\                     /\/\ \n'
# printf '  \/\/            {  "---"  }                   \/\/ \n'
# printf '  /\/\            {  O   O  }                   /\/\ \n'
# printf '  \/\/            ~~>  V  <~~                   \/\/ \n'
# printf '  /\/\               \    /~~~~                 /\/\ \n'
# printf '  \/\/              /     \    \_               \/\/ \n'
# printf '  /\/\             {       }\  )_\_   _         /\/\ \n'
# printf '  \/\/             |  \_/  |/ /  \_\_/ )        \/\/ \n'
# printf '  /\/\              \__/  /(_/     \__/         /\/\ \n'
# printf '  \/\/                (  /                      \/\/ \n'
# printf '  /\/\                 \/                       /\/\ \n'
# printf '  \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ \n'
# printf '  \//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ \n'
printf '                     NOW RUNNING F71 CONVERSIONS \n'
# printf " +--------------------------------------------------------------------+ \n "
# printf " | -   -   -   -      -     -   -         -   - -      -  -    -      | \n "
# printf " |    - -     -   -   -  -           -       -   .---.  _  - -    -   | \n "
# printf " | -       -  { ! }   -     -     -      -      /_   .-{_}   - -   -  | \n "
# printf " |  - -  .--. /'    -   -       -   -        -  }u\ /  - -    -   -   | \n "
# printf " |-  .-./ ___\  -   ||     -   -      -  -      \-_/_  -     -     -  | \n "
# printf " | - '-'\|o o|   -  ||   -  -      -         ___/    \   -      -    -| \n "
# printf " | - -  __\c/__ -  .-. -  -    -        -  {\___/ /   \  -   -    -   | \n "
# printf " |  - .'  \:/  \. /\/|  -     -       -   / {_}__/ \   \  -    -   -  | \n "
# printf " |-  / /{  :  }\ V /||         -   -     / /   -    \_.'\    -    -  -| \n "
# printf " |   \_\{  :  } '-' ||  - -   -      ___/ /  -   -  /  /| -    -   -  | \n "
# printf " | -  {_{__:__}   __||___ -  -  {_.-' __\/-   -    /  / |   -     -   | \n "
# printf " |-  -  \  |  /  /      /     {   }.-'   |   -  - {  {\  \   -    - - | \n "
# printf " +--------------------------------------------------------------------+ \n"


# Check if all four arguments are passed
if [ "$#" -ne 7 ]; then
  echo 0: $0
  echo 1: $1
  echo 2: $2
  echo 3: $3
  echo 4: $4
  echo 5: $5
  echo 6: $6
  echo 7: $7
  echo 8: $8
  echo "Usage: $0 <materialNumber> <temperature> <filename> <dictionaryFilename> <outputFilename> <scaleVersion> <obiwanPath>"
  exit 1
fi

# Assign the arguments to variables
materialNumber=$1
temperature=$2
filename=$3
dictionaryFilename=$4
outputFilename=$5
scaleVersion=$6
obiwanPath=$7

# Function to load the dictionary into an array
load_dictionary() {
  if [ ! -f "$1" ]; then
    echo "Error: Dictionary file '$1' not found!"
    exit 1
  fi

  # Read the dictionary file into an array
  mapfile -t dictionary < "$1"

  # Trim spaces from each entry in the dictionary
  for i in "${!dictionary[@]}"; do
    dictionary[$i]="${dictionary[$i]// /}"  # Remove all spaces from the entry
  done
}

# Function to check if a nuclide is in the dictionary
is_in_dictionary() {
  for entry in "${dictionary[@]}"; do
    if [[ "$entry" == "$1" ]]; then
      return 0
    fi
  done
  return 1
}

# Bash function that reads data from the file
grab_data() {
  # Check if the file exists
  if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found!"
    exit 1
  fi

  # Read the file line by line
  if [[ "$scaleVersion" == "631" || "$scaleVersion" == "632"  || "$scaleVersion" == "700" ]]; then
    echo "Handling obiwan call for SCALE 6.3.1 or 6.3.2 or 7.0.0"
    echo obiwan path is ${obiwanPath}
    ${obiwanPath} view -format=csv -units=atom -prec=8 -idform='{:ee}-{:A}{:m}' "$1"
  elif [[ "$scaleVersion" == "624" ]]; then
    echo "Handling obiwan call for SCALE 6.2.4"
    echo obiwan path is ${obiwanPath}
    ${obiwanPath} view -format=csv -units=atom -prec=8 -idform='$ee-$A$m' "$1"
  else
    echo "Error: Unsupported scaleVersion '$scaleVersion'" >&2
    exit 1
  fi
}

# Load and trim the dictionary
load_dictionary "$dictionaryFilename"

echo STARTING LINE
echo "' material number $materialNumber in the stdcmp format with filename $outputFilename" >> $outputFilename
# Process the output of the function and format the result
grab_data "$filename" | while IFS=, read -r nuclide atomdensity; do
  # Trim spaces from nuclide and atomdensity
  nuclide="${nuclide// /}"  # Remove all spaces from the nuclide
  atomdensity="${atomdensity// /}"  # Remove all spaces from the atomdensity

CUTOFF="1e-16"
if is_in_dictionary "$nuclide" && \
   awk -v val="$atomdensity" -v cut="$CUTOFF" 'BEGIN { exit !(val >= cut) }'
then
  echo "${nuclide} $materialNumber 0 ${atomdensity} $temperature end"
  echo "${nuclide} $materialNumber 0 ${atomdensity} $temperature end" >> "$outputFilename"
fi
done
echo ENDING LINE
