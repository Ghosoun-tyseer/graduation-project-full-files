import pandas as pd
import torch as th
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import os

#استيراد أدوات HuggingFace
#AutoTokenizer → يحول الـ SMILES إلى Tokens يفهمها النموذج.
#AutoModel → يحمل نموذج ChemBERTa نفسه.
def generate_chemberta_embeddings(input_csv, output_pkl):
    # 1. إعداد النموذج (استخدام نسخة ChemBERTa-768 المفتوحة)
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)#تحميل الـ Tokenizer.
    model = AutoModel.from_pretrained(model_name) #تحميل أوزان ChemBERTa
    model.eval()  #وضع النموذج في وضع التقييم Evaluation Mode.
    # لا يوجد تدريب ولا تحديث لاوزان فقط استخراج ميزات 

    # 2. قراءة ملف SMILES
    df = pd.read_csv(input_csv)
    # التأكد من وجود عمود SMILES
    if 'SMILES' not in df.columns:
        raise ValueError("الملف يجب أن يحتوي على عمود باسم SMILES")

    embeddings_dict = {} # انشاء مكان حفظ النتائج 

    print("جاري تحويل الـ SMILES إلى إيمبدنج...")
    with th.no_grad(): # ايقاف حساب ال Gradients بما اننا لا ندرب النموذج لتوفير وقت وذاكره 
        for index, row in tqdm(df.iterrows(), total=len(df)): #المرور على جميع الأدوية
            smiles = str(row['SMILES'])
            drug_id = row['ID'] # نستخدم الـ ID كـ Key
            
            # تحويل النص إلى Tokens
            #تتحول السمايل الى :tensor([[15,15,23]])   مثلا 
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)#return_tensors="pt"  إرجاع النتيجة كـ PyTorch Tensor.
            # استخراج الميزات من النموذج
            #لكل توكنيكون متجه 768 رقم 
            outputs = model(**inputs)
            # نأخذ المتوسط (Mean Pooling) لتمثيل الجزيء بالكامل في ناقل واحد
           #يحسب المتوسط لكل رقم ) مجموعهم على عددهم
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            # يربط الايمبدنج بالID
            embeddings_dict[drug_id] = emb

    # 3. حفظ النتيجة في مجلد feat
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"تم بنجاح! الملف محفوظ في: {output_pkl}")

if __name__ == "__main__":
    #   المسار  للملفات
    generate_chemberta_embeddings('../data/Cdataset/drug.csv', '../feat/Cdataset/ChemBERTa_drug_emb.pkl')