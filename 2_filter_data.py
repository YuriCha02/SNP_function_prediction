import json
from time import sleep

with open('snps.json', 'r') as f:
    snps = json.load(f)

# Initialize an empty list for the filtered SNPs
filtered_snps = []

# Iterate over each snp in the snps list
for snp in snps:
    # Initialize a new dictionary for the filtered snp
    filtered_snp = {}

    # Extract necessary keys
    filtered_snp['rsId'] = snp['rsId']
    filtered_snp['functionalClass'] = snp['functionalClass']

    # Initialize an empty list for the genomicContexts
    filtered_snp['genomicContexts'] = []

    # Iterate over each context in the genomicContexts list
    for context in snp['genomicContexts']:
        # Initialize new dictionary for the context
        filtered_context = {}

        # Extract necessary keys
        filtered_context['isIntergenic'] = context['isIntergenic']
        filtered_context['isUpstream'] = context['isUpstream']
        filtered_context['isDownstream'] = context['isDownstream']
        filtered_context['distance'] = context['distance']
        filtered_context['location'] = {
            "chromosomeName": context['location']["chromosomeName"],
            "chromosomePosition": context['location']["chromosomePosition"]
        }

        # Append the filtered context to the list
        filtered_snp['genomicContexts'].append(filtered_context)

    # Append the filtered snp to the list
    filtered_snps.append(filtered_snp)

# Save to a new JSON file
with open('filtered_snps.json', 'w') as outfile:
    json.dump(filtered_snps, outfile, indent=4)