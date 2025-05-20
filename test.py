# PFAS Transporter Identification Pipeline
# Author: [Your Name]
# Description: Pipeline to identify likely PFAS importers in E. coli based on ligand similarity analysis.

# Required libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import PandasTools
from sklearn.linear_model import LogisticRegression

# Step 1: Retrieve transporter data
# Data obtained from: https://www.membranetransport.org/transportDB2/complete_list_v2.html
ecoli_transporters = pd.read_csv('ecoli536.csv')
# Expected Columns: ['gene_name', 'annotation', 'consensus_sequence', 'ligand_smiles']

# Step 2: Prepare PFAS reference molecules
# Example PFAS SMILES strings (replace with desired PFAS molecules)
pfas_smiles = [
    "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", # Example long-chain PFAS
    "FC(F)(F)C(F)(F)C(F)(F)F"          # Example short-chain PFAS
]
pfas_mols = [Chem.MolFromSmiles(smile) for smile in pfas_smiles]
pfas_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in pfas_mols]

# Step 3: Calculate ligand chemical similarity
# Define a function to calculate chemical similarity
def calculate_similarity(ligand_smiles, reference_fps):
    ligand_mol = Chem.MolFromSmiles(ligand_smiles)
    if ligand_mol is None:
        return np.nan
    ligand_fp = AllChem.GetMorganFingerprintAsBitVect(ligand_mol, radius=2, nBits=2048)
    similarities = [DataStructs.TanimotoSimilarity(ligand_fp, ref_fp) for ref_fp in reference_fps]
    return np.mean(similarities)

# Apply chemical similarity calculation
ecoli_transporters['similarity_score'] = ecoli_transporters['ligand_smiles'].apply(lambda x: calculate_similarity(x, pfas_fps))

# Step 4: Exploratory Data Analysis
similarity_distribution = ecoli_transporters['similarity_score'].describe()
print(similarity_distribution)

# Step 5: Logistic Regression Model (classification into likely/unlikely importer)
# Prepare data
ecoli_transporters.dropna(subset=['similarity_score'], inplace=True)
ecoli_transporters['importer_label'] = ecoli_transporters['annotation'].apply(lambda x: 1 if 'importer' in x.lower() else 0)
X = ecoli_transporters[['similarity_score']]
y = ecoli_transporters['importer_label']

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities of being an importer
ecoli_transporters['importer_probability'] = model.predict_proba(X)[:, 1]

# Rank transporters by probability
ecoli_transporters.sort_values('importer_probability', ascending=False, inplace=True)

# Step 6: Output final ranked results
final_results = ecoli_transporters[['gene_name', 'similarity_score', 'annotation', 'importer_probability']]
final_results.to_csv('ranked_pfas_importers.csv', index=False)

print("Top likely PFAS importers:")
print(final_results.head(10))
