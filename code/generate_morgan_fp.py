import pandas as pd
import numpy as np
from rdkit import Chem   #تستخدم لتحويل SMILES إلى Moleculer graph.
from rdkit.Chem import AllChem #فيها خوارزمية Morgan fingerprint
import os

def smiles_to_morgan(smiles, radius=3, n_bits=2048):
    """تحويل صيغة SMILES إلى Morgan Fingerprint"""
    try:
        # التأكد من أن الصيغة ليست فارغة
        if pd.isna(smiles) or smiles == "":
            return np.zeros(n_bits)# [0,0,0,0,0,0,...]
        
        mol = Chem.MolFromSmiles(smiles)#تحويل SMILES إلى Molecule
        if mol is not None:
            # توليد البصمة الكيميائية (نصف القطر 2 يعادل ECFP4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp) # يحول fingerprint من RDKit bit vector إلى numpy array (0/1)
        else:
            return np.zeros(n_bits) 
    except:
        return np.zeros(n_bits) #حماية الكود من:SMILES corrupt او unexpected RDKit errors

input_file = os.path.join(os.path.dirname(__file__), "..", "data", "Cdataset", "drug.csv")
output_dir = os.path.join(os.path.dirname(__file__), "..", "feat", "Cdataset")
# 2. تحميل البيانات
print(f"Reading file from: {input_file}")
df_drug = pd.read_csv(input_file)

# 3. استخراج البيانات باستخدام أسماء الأعمدة  
drug_ids = df_drug['Drug'].values    # المعرف (مثل DB0001)
smiles_list = df_drug['SMILES'].values # الصيغة الكيميائية

# 4. توليد البصمات (Fingerprints)
# print("Generating Morgan Fingerprints (1024 bits)...")

print("Generating Morgan Fingerprints (Radius=3, 2048 bits)...")

drug_to_fp = {}
for d_id, smiles in zip(drug_ids, smiles_list):
    drug_to_fp[d_id] = smiles_to_morgan(smiles)

# 5. حفظ النتيجة بصيغة pkl
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#output_file = os.path.join(output_dir, "Morgan_drug_emb.pkl")

output_file = os.path.join(output_dir, "Morgan_R3_2048_drug_emb.pkl")
pd.to_pickle(drug_to_fp, output_file)

print("-" * 30)
print(f"Success! Processed {len(drug_to_fp)} drugs.")
print(f"Saved to: {output_file}")