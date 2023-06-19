import requests
import json
from time import sleep
from apscheduler.schedulers.blocking import BlockingScheduler

# Receive data through OpenAP
def request_data():
    snps = []

    # Load data from chromosome 1 to 22
    for i in range(1, 23, 1):
        url = f'https://www.ebi.ac.uk/gwas/rest/api/snpLocation/{i}:0-2000000000'
        SNP = requests.get(url)
        snp_json = SNP.json()
        for snp in snp_json:
            snps.append(snp['_embedded']['singleNucleotidePolymorphisms'])
        print("chromosome", i, "is done.")
        sleep(30)

    with open('snps.json', 'w') as f:
        json.dump(snps, f)

# Schedule requesting data
scheduler = BlockingScheduler({'apscheduler.timezone':'UTC'})
scheduler.add_job(request_data, 'date', run_date='2023-06-16 08:00:00')
    
