#app.py
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from torch import nn
from mil import BagModel

# Load label encoders from training models
le_chromosomeNames = torch.load('label_encoder_chromosomeNames.pth')
le_functionalClass = torch.load('label_encoder_functionalClass.pth')

# Define custom prepNN
prepNN = torch.nn.Sequential(
    torch.nn.Linear(6, 64),  # Input layer
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32), # 1 hidden layer
    torch.nn.ReLU(),
)

# Define custom afterNN
afterNN = torch.nn.Sequential(
    torch.nn.Linear(32, 24),  # Output layer
    torch.nn.LogSoftmax(dim=1) 
)

# Classes for prediction
classes = ['intron_variant', 'intergenic_variant', 'regulatory_region_variant', 'non_coding_transcript_exon_variant', 
               '3_prime_UTR_variant', 'missense_variant', 'upstream_gene_variant', 'downstream_gene_variant', 
               'synonymous_variant', '5_prime_UTR_variant', 'TF_binding_site_variant', 'splice_region_variant', 
               'stop_gained', 'frameshift_variant', 'splice_donor_variant', 'splice_acceptor_variant', 'inframe_deletion', 
               'inframe_insertion', 'stop_lost', 'start_lost', 'mature_miRNA_variant', 'rare_function', 'protein_altering_variant', 'intron']

# Define model with prepNN, afterNN and torch.mean as aggregation function
model = BagModel(prepNN, afterNN, torch.mean)

# Load the trained model
model.load_state_dict(torch.load('model.pth'))
model.eval()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html'), 200

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        numContexts = int(request.form.get('numContexts'))

        genomicContexts = []

        for i in range(numContexts):
            isIntergenic = request.form.get(f'isIntergenic{i}') == 'True'
            isUpstream = request.form.get(f'isUpstream{i}') == 'True'
            isDownstream = request.form.get(f'isDownstream{i}') == 'True'
            distance = request.form.get(f'distance{i}')
            chromosomeName = request.form.get(f'chromosomeName{i}')
            chromosomePosition = request.form.get(f'chromosomePosition{i}')

            genomicContexts.append({
                'isIntergenic': isIntergenic,
                'isUpstream': isUpstream,
                'isDownstream': isDownstream,
                'distance': distance,
                'location': {
                    'chromosomeName': chromosomeName,
                    'chromosomePosition': chromosomePosition
                }
            })

        features_list = []
        
        for context in genomicContexts:
            # Extract features from the request data
            features = torch.tensor([
                float(context['isIntergenic']), 
                float(context['isUpstream']), 
                float(context['isDownstream']), 
                float(context['distance']), 
                float(le_chromosomeNames.transform([context['location']['chromosomeName']])[0]),
                float(context['location']['chromosomePosition'])
                ]).unsqueeze(0)

            features_list.append(features)

        features_tensor = torch.cat(features_list, dim=0)
        features_tensor = features_tensor.float()

        # Set inner_ids to zeros to indicate that all contexts belong to the same bag
        inner_ids = torch.zeros(numContexts, dtype=torch.long)

        with torch.no_grad():
            log_probs = model((features_tensor, inner_ids))
            # Convert to probabilities
            probs = torch.exp(log_probs)
            # Get top 5 predictions
            top_probs, top_idxs = probs.topk(5)
            top_idxs = top_idxs.view(-1) #Change the shape to 1d array
            top_classes = le_functionalClass.inverse_transform(top_idxs.tolist())

        return jsonify({'Top 5 predictions': list(zip(top_classes))})


@app.route('/available_chromosomes', methods=['GET'])
def available_chromosomes():
    options = ('1', 'CHR_HSCHR1_3_CTG31', 'CHR_HSCHR1_9_CTG3', 'CHR_HG2095_PATCH', 'CHR_HSCHR1_3_CTG32_1', 'CHR_HG986_PATCH', 
               'CHR_HSCHR1_1_CTG3', 'CHR_HSCHR1_1_CTG32_1', 'CHR_HSCHR1_5_CTG32_1', 'CHR_HSCHR1_2_CTG31', 'CHR_HSCHR1_1_CTG31', 
               'CHR_HG2058_PATCH', 'CHR_HG2002_PATCH', 'CHR_HG1832_PATCH', 'CHR_HSCHR1_4_CTG3', 'CHR_HSCHR1_6_CTG3', 'CHR_HSCHR1_1_CTG11', 
               'CHR_HSCHR1_3_CTG3', 'CHR_HSCHR1_8_CTG3', 'CHR_HG2104_PATCH', 'CHR_HG1342_HG2282_PATCH', 'CHR_HSCHR1_4_CTG32_1', '2', 
               'CHR_HG2232_PATCH', 'CHR_HSCHR2_8_CTG7_2', 'CHR_HSCHR2_1_CTG7_2', 'CHR_HSCHR2_6_CTG7_2', 'CHR_HG2233_PATCH', 
               'CHR_HSCHR2_1_CTG5', 'CHR_HSCHR2_4_CTG1', 'CHR_HSCHR2_2_CTG7_2', 'CHR_HSCHR2_1_CTG15', 'CHR_HSCHR2_2_CTG15', 
               'CHR_HSCHR2_2_CTG1', 'CHR_HSCHR2_1_CTG7', 'CHR_HSCHR2_3_CTG15', 'CHR_HSCHR2_1_CTG1', 'CHR_HG2290_PATCH', 
               'CHR_HSCHR2_4_CTG7_2', 'CHR_HSCHR2_3_CTG7_2', 'CHR_HSCHR2_7_CTG7_2', 'CHR_HSCHR2_5_CTG7_2', '3', 'CHR_HSCHR3_1_CTG1', 
               'CHR_HG2236_PATCH', 'CHR_HG126_PATCH', 'CHR_HG2235_PATCH', 'CHR_HSCHR3_5_CTG2_1', 'CHR_HSCHR3_4_CTG2_1', 
               'CHR_HSCHR3_3_CTG1', 'CHR_HSCHR3_4_CTG1', 'CHR_HSCHR3_5_CTG3', 'CHR_HSCHR3_7_CTG3', 'CHR_HSCHR3_4_CTG3', 
               'CHR_HSCHR3_1_CTG3', 'CHR_HSCHR3_6_CTG3', 'CHR_HSCHR3_3_CTG3', 'CHR_HSCHR3_8_CTG3', 'CHR_HSCHR3_2_CTG2_1', 
               'CHR_HSCHR3_2_CTG3', 'CHR_HSCHR3_3_CTG2_1', 'CHR_HG2066_PATCH', 'CHR_HSCHR3_1_CTG2_1', 'CHR_HSCHR3_9_CTG3', 
               '4', 'CHR_HSCHR4_3_CTG12', 'CHR_HSCHR4_6_CTG12', 'CHR_HSCHR4_7_CTG12', 'CHR_HSCHR4_12_CTG12', 'CHR_HSCHR4_2_CTG12', 
               'CHR_HSCHR4_2_CTG4', 'CHR_HSCHR4_1_CTG4', 'CHR_HSCHR4_1_CTG9', 'CHR_HSCHR4_1_CTG12', 'CHR_HSCHR4_9_CTG12', 
               'CHR_HSCHR4_1_CTG6', 'CHR_HSCHR4_5_CTG12', 'CHR_HSCHR4_1_CTG8_1', 'CHR_HSCHR4_8_CTG12', 'CHR_HSCHR4_4_CTG12', '5', 
               'CHR_HG30_PATCH', 'CHR_HSCHR5_3_CTG1', 'CHR_HSCHR5_6_CTG1', 'CHR_HSCHR5_4_CTG1_1', 'CHR_HSCHR5_5_CTG1', 'CHR_HSCHR5_4_CTG1', 
               'CHR_HSCHR5_2_CTG1_1', 'CHR_HSCHR5_1_CTG1_1', 'CHR_HSCHR5_2_CTG1', 'CHR_HSCHR5_1_CTG5', 'CHR_HSCHR5_9_CTG1', 
               'CHR_HSCHR5_3_CTG1_1', 'CHR_HSCHR5_2_CTG5', 'CHR_HSCHR5_3_CTG5', 'CHR_HSCHR5_8_CTG1', 'CHR_HSCHR5_7_CTG1', 
               'CHR_HSCHR5_1_CTG1', '6', 'CHR_HSCHR6_MHC_MANN_CTG1', 'CHR_HSCHR6_MHC_MCF_CTG1', 'CHR_HSCHR6_MHC_DBB_CTG1', 
               'CHR_HSCHR6_MHC_QBL_CTG1', 'CHR_HSCHR6_MHC_SSTO_CTG1', 'CHR_HSCHR6_MHC_COX_CTG1', 'CHR_HSCHR6_MHC_APD_CTG1', 
               'CHR_HSCHR6_8_CTG1', 'CHR_HSCHR6_1_CTG6', 'CHR_HSCHR6_1_CTG8', 'CHR_HSCHR6_1_CTG2', 'CHR_HSCHR6_1_CTG7', 
               'CHR_HG2128_PATCH', 'CHR_HSCHR6_1_CTG5', 'CHR_HSCHR6_1_CTG10', 'CHR_HG2057_PATCH', 'CHR_HSCHR6_1_CTG3', 
               'CHR_HG2072_PATCH', 'CHR_HSCHR6_1_CTG9', 'CHR_HSCHR6_1_CTG4', 'CHR_HG1651_PATCH', '7', 'CHR_HSCHR7_3_CTG4_4', 
               'CHR_HSCHR7_1_CTG7', 'CHR_HG2266_PATCH', 'CHR_HG708_PATCH', 'CHR_HSCHR7_2_CTG6', 'CHR_HSCHR7_2_CTG7', 'CHR_HG2088_PATCH', 
               'CHR_HG2239_PATCH', 'CHR_HSCHR7_2_CTG1', 'CHR_HSCHR7_1_CTG1', 'CHR_HSCHR7_3_CTG6', 'CHR_HSCHR7_1_CTG4_4', 
               'CHR_HSCHR7_2_CTG4_4', 'CHR_HSCHR7_1_CTG6', '22', '8', 'CHR_HSCHR8_3_CTG7', 'CHR_HG76_PATCH', 'CHR_HSCHR8_7_CTG1', 
               'CHR_HSCHR8_9_CTG1', 'CHR_HSCHR8_8_CTG1', 'CHR_HSCHR8_5_CTG7', 'CHR_HSCHR8_1_CTG1', 'CHR_HSCHR8_5_CTG1', 'CHR_HSCHR8_3_CTG1', 
               'CHR_HSCHR8_2_CTG1', 'CHR_HG2068_PATCH', 'CHR_HG2067_PATCH', 'CHR_HSCHR8_1_CTG6', 'CHR_HSCHR8_1_CTG7', 'CHR_HSCHR8_6_CTG7', 
               'CHR_HG2419_PATCH', 'CHR_HSCHR8_6_CTG1', 'CHR_HSCHR8_4_CTG7', 'CHR_HSCHR8_4_CTG1', 'CHR_HSCHR8_2_CTG7', '9', 
               'CHR_HSCHR9_1_CTG4', 'CHR_HG2030_PATCH', 'CHR_HSCHR9_1_CTG6', 'CHR_HSCHR9_1_CTG1', 'CHR_HSCHR9_1_CTG2', 'CHR_HSCHR9_1_CTG5', 
               'CHR_HSCHR9_1_CTG7', 'CHR_HSCHR9_1_CTG3', '10', 'CHR_HSCHR10_1_CTG2', 'CHR_HG2334_PATCH', 'CHR_HG2191_PATCH', 
               'CHR_HSCHR10_1_CTG4', 'CHR_HSCHR10_1_CTG1', 'CHR_HSCHR10_1_CTG6', '11', 'CHR_HG2116_PATCH', 'CHR_HSCHR11_1_CTG3', 
               'CHR_HSCHR11_1_CTG7', 'CHR_HSCHR11_1_CTG8', 'CHR_HSCHR11_3_CTG1', 'CHR_HSCHR11_2_CTG1', 'CHR_HG107_PATCH', 
               'CHR_HG142_HG150_NOVEL_TEST', 'CHR_HSCHR11_1_CTG1_1', 'CHR_HSCHR11_1_CTG2', 'CHR_HSCHR11_1_CTG5', 'CHR_HSCHR11_1_CTG1_2', 
               'CHR_HG151_NOVEL_TEST', 'CHR_HG2217_PATCH', 'CHR_HSCHR11_1_CTG6', 'CHR_HSCHR11_2_CTG1_1', 'CHR_HG1708_PATCH', '12', 
               'CHR_HSCHR12_1_CTG2', 'CHR_HG1815_PATCH', 'CHR_HG2247_PATCH', 'CHR_HSCHR12_6_CTG2_1', 'CHR_HSCHR12_1_CTG1', 
               'CHR_HG1362_PATCH', 'CHR_HSCHR12_2_CTG2_1', 'CHR_HSCHR12_2_CTG2', 'CHR_HSCHR12_3_CTG2', 'CHR_HSCHR12_4_CTG2_1', 
               'CHR_HSCHR12_4_CTG2', 'CHR_HG2063_PATCH', 'CHR_HSCHR12_8_CTG2_1', 'CHR_HSCHR12_3_CTG2_1', 'CHR_HSCHR12_1_CTG2_1', 
               'CHR_HG23_PATCH', 'CHR_HG2047_PATCH', 'CHR_HSCHR12_5_CTG2_1', 'CHR_HSCHR12_7_CTG2_1', 'CHR_HSCHR12_2_CTG1', '13', 
               'CHR_HSCHR13_1_CTG3', 'CHR_HSCHR13_1_CTG8', 'CHR_HG2288_HG2289_PATCH', 'CHR_HSCHR13_1_CTG1', 'CHR_HSCHR13_1_CTG2', 
               'CHR_HSCHR13_1_CTG4', 'CHR_HSCHR13_1_CTG5', 'CHR_HSCHR13_1_CTG7', 'CHR_HG2249_PATCH', 'CHR_HG2216_PATCH', 
               'CHR_HSCHR13_1_CTG6', 'CHR_HG2291_PATCH', '14', 'CHR_HG1_PATCH', 'CHR_HSCHR14_7_CTG1', 'CHR_HSCHR14_3_CTG1', 
               'CHR_HSCHR14_8_CTG1', 'CHR_HSCHR14_1_CTG1', 'CHR_HSCHR14_2_CTG1', '15', 'CHR_HSCHR15_4_CTG8', 'CHR_HSCHR15_6_CTG8', 
               'CHR_HSCHR15_1_CTG8', 'CHR_HSCHR15_5_CTG8', 'CHR_HSCHR15_2_CTG8', 'CHR_HSCHR15_3_CTG8', 'CHR_HSCHR15_1_CTG3', 
               'CHR_HSCHR15_3_CTG3', 'CHR_HSCHR15_2_CTG3', '16', 'CHR_HSCHR16_1_CTG1', 'CHR_HSCHR16_5_CTG3_1', 'CHR_HSCHR16_3_CTG3_1', 
               'CHR_HSCHR16_1_CTG3_1', 'CHR_HSCHR16_CTG2', 'CHR_HSCHR16_3_CTG1', 'CHR_HG926_PATCH', 'CHR_HSCHR16_4_CTG3_1', 
               'CHR_HSCHR16_4_CTG1', 'CHR_HSCHR16_5_CTG1', 'CHR_HSCHR16_2_CTG3_1', '17', 'CHR_HSCHR17_2_CTG5', 'CHR_HSCHR17_1_CTG5', 
               'CHR_HSCHR17_7_CTG4', 'CHR_HG2046_PATCH', 'CHR_HSCHR17_1_CTG1', 'CHR_HSCHR17_4_CTG4', 'CHR_HSCHR17_2_CTG4', 
               'CHR_HG2285_HG106_HG2252_PATCH', 'CHR_HSCHR17_1_CTG2', 'CHR_HSCHR17_1_CTG9', 'CHR_HSCHR17_2_CTG2', 'CHR_HSCHR17_3_CTG4', 
               'CHR_HSCHR17_3_CTG1', 'CHR_HSCHR17_10_CTG4', 'CHR_HSCHR17_5_CTG4', 'CHR_HSCHR17_6_CTG4', 'CHR_HSCHR17_1_CTG4', 
               'CHR_HSCHR17_3_CTG2', 'CHR_HSCHR17_11_CTG4', 'CHR_HSCHR17_9_CTG4', 'CHR_HSCHR17_8_CTG4', 'CHR_HSCHR17_2_CTG1', '18', 
               'CHR_HG2442_PATCH', 'CHR_HSCHR18_3_CTG2_1', 'CHR_HG2213_PATCH', 'CHR_HSCHR18_ALT21_CTG2_1', 'CHR_HSCHR18_1_CTG2_1', 
               'CHR_HSCHR18_1_CTG1', 'CHR_HSCHR18_2_CTG2_1', 'CHR_HSCHR18_ALT2_CTG2_1', 'CHR_HSCHR18_5_CTG1_1', 'CHR_HSCHR18_1_CTG1_1', 
               'CHR_HSCHR18_2_CTG1_1', 'CHR_HSCHR18_2_CTG2', 'CHR_HSCHR18_1_CTG2', 'CHR_HSCHR18_4_CTG1_1', '19', 
               'CHR_HSCHR19LRC_LRC_J_CTG3_1', 'CHR_HSCHR19LRC_LRC_S_CTG3_1', 'CHR_HSCHR19LRC_COX1_CTG3_1', 'CHR_HSCHR19_4_CTG3_1', 
               'CHR_HSCHR19LRC_PGF1_CTG3_1', 'CHR_HSCHR19LRC_LRC_T_CTG3_1', 'CHR_HSCHR19LRC_LRC_I_CTG3_1', 'CHR_HSCHR19LRC_PGF2_CTG3_1', 
               'CHR_HSCHR19LRC_COX2_CTG3_1', 'CHR_HSCHR19_1_CTG3_1', 'CHR_HSCHR19_4_CTG2', 'CHR_HSCHR19_1_CTG2', 'CHR_HSCHR19_2_CTG2', 
               'CHR_HG26_PATCH', 'CHR_HSCHR19_3_CTG3_1', 'CHR_HSCHR19_3_CTG2', 'CHR_HSCHR19KIR_FH05_A_HAP_CTG3_1', 
               'CHR_HSCHR19KIR_GRC212_AB_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH15_B_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH05_B_HAP_CTG3_1', 
               'CHR_HSCHR19KIR_FH06_BA1_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH08_A_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH06_A_HAP_CTG3_1', 
               'CHR_HSCHR19KIR_CA04_CTG3_1', 'CHR_HSCHR19KIR_ABC08_A1_HAP_CTG3_1', 'CHR_HSCHR19KIR_HG2393_CTG3_1', 
               'CHR_HSCHR19KIR_FH15_A_HAP_CTG3_1', 'CHR_HSCHR19KIR_T7526_A_HAP_CTG3_1', 'CHR_HSCHR19KIR_ABC08_AB_HAP_T_P_CTG3_1', 
               'CHR_HSCHR19KIR_LUCE_A_HAP_CTG3_1', 'CHR_HSCHR19KIR_G085_BA1_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH13_A_HAP_CTG3_1', 
               'CHR_HSCHR19KIR_T7526_BDEL_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH13_BA2_HAP_CTG3_1', 'CHR_HSCHR19KIR_FH08_BAX_HAP_CTG3_1', 
               'CHR_HSCHR19KIR_RSH_A_HAP_CTG3_1', 'CHR_HSCHR19KIR_LUCE_BDEL_HAP_CTG3_1', 'CHR_HSCHR19KIR_G085_A_HAP_CTG3_1', 
               'CHR_HSCHR19KIR_G248_A_HAP_CTG3_1', 'CHR_HSCHR19KIR_GRC212_BA1_HAP_CTG3_1', 'CHR_HSCHR19KIR_HG2396_CTG3_1', 
               'CHR_HSCHR19KIR_RSH_BA2_HAP_CTG3_1', 'CHR_HSCHR19KIR_G248_BA2_HAP_CTG3_1', 'CHR_HSCHR19KIR_HG2394_CTG3_1', 
               'CHR_HSCHR19KIR_RP5_B_HAP_CTG3_1', 'CHR_HSCHR19KIR_ABC08_AB_HAP_C_P_CTG3_1', 'CHR_HSCHR19_5_CTG2', 'CHR_HG2021_PATCH', 
               'CHR_HSCHR19_2_CTG3_1', '20', 'CHR_HSCHR20_1_CTG2', 'CHR_HSCHR20_1_CTG1', 'CHR_HSCHR20_1_CTG3', 'CHR_HSCHR20_1_CTG4', '21', 
               'CHR_HSCHR21_1_CTG1_1', 'CHR_HSCHR21_4_CTG1_1', 'CHR_HSCHR21_2_CTG1_1', 'CHR_HSCHR21_3_CTG1_1', 'CHR_HSCHR21_8_CTG1_1', 
               'CHR_HSCHR21_6_CTG1_1', 'CHR_HSCHR21_5_CTG2', 'CHR_HSCHR22_1_CTG2', 'CHR_HSCHR22_1_CTG5', 'CHR_HG1311_PATCH', 
               'CHR_HSCHR22_1_CTG7', 'CHR_HSCHR22_1_CTG4', 'CHR_HSCHR22_3_CTG1', 'CHR_HSCHR22_5_CTG1', 'CHR_HSCHR22_7_CTG1', 
               'CHR_HSCHR22_2_CTG1', 'CHR_HSCHR22_8_CTG1', 'CHR_HSCHR22_1_CTG6', 'CHR_HSCHR22_4_CTG1', 'CHR_HSCHR22_6_CTG1', 
               'CHR_HSCHR22_1_CTG1', 'CHR_HSCHR22_1_CTG3')

    return jsonify({'Avaiable options': options}), 200


if __name__ == '__main__':
    app.run(port=5000)