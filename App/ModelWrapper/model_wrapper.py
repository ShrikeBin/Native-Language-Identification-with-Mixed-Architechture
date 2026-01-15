import torch
import shap
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel
import importlib

# ===== Device =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Train Model Maps =====
model_maps = {
    'DistilBERT': "distilbert-base-uncased",
    'RoBERTa': "roberta-base",
    'RoBERTaLarge': "roberta-large",
    'DeBERTa': "microsoft/deberta-v3-large",
    'MpNet': "sentence-transformers/all-mpnet-base-v2",
}

# ===== Trait Label Maps =====
label_maps = {
    'gender': {0: "female", 1: "male"},
    'political': {0: "left", 1: "center", 2: "right"},
    'mbti': {0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"},
    'language': {0: "English", 1: "German", 2: "Nordic", 3: "French", 4: "Italian", 5: "Portuguese", 6: "Spanish", 7: "Russian", 8: "Polish", 9: "Other Slavic", 10: "Turkic", 11: "Chinese", 12: "Vietnamese", 13: "Koreanic", 14: "Japonic", 15: "Tai", 16: "Indonesian", 17: "Uralic", 18: "Arabic", 19: "Indo-Iranian"},
}

# ===== Model Wrapper =====
class Model:
    def __init__(self, trait_name, head_type, model='DistilBERT', train='Full'):

        # === Basic Config ===
        self.name = f"{trait_name} ({model} {head_type} {train})"
        self.type = head_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_maps[model])
        self.label_map = label_maps.get(trait_name, None)

        # === Model Base ===
        path = f"Training/MODELS/{trait_name}/{model}{head_type}{train}/model"
        
        if head_type == 'Classification': # Transformers Classifiers
            self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=len(self.label_map))
        else: # Custom Head
            head_module_path = f"App.ModelWrapper.custom_heads.{head_type}"
            match head_type:
                case 'Regression':
                    head_module = importlib.import_module(head_module_path)
                    CustomRegression = getattr(head_module, "CustomRegression")
                    self.model = CustomRegression(model_maps[model])
                case 'MixedCNN':
                    head_module = importlib.import_module(f"{head_module_path}.{model}")
                    MixedClassifier = getattr(head_module, "MixedClassifier")
                    model_path = f"Training/MODELS/{trait_name}/{model}ClassificationFull/model"
                    cnn_path = f"Training/MODELS/{trait_name}/{model}{head_type}{train}/CNN/model.safetensors"
                    self.model = MixedClassifier(model_path, cnn_path, num_classes=len(self.label_map))
                case 'CNN':
                    head_module = importlib.import_module(head_module_path)
                    CustomCNN = getattr(head_module, "CustomCNN")
                    self.model = CustomCNN(num_classes=len(self.label_map))
                    self.name = f"{trait_name} ({head_type})"
                    self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                    path = f"Training/MODELS/{trait_name}/{head_type}/model"
                case 'TEST':
                    # THIS HERE IS TEMPORARY FOR TESTING CUSTOM DATESETS
                    # TODO: REMOVE LATER, or INTEGRATE PROPERLY
                    head_module = importlib.import_module(f"App.ModelWrapper.custom_heads.CNN")
                    CustomCNN = getattr(head_module, "CustomCNN")
                    self.model = CustomCNN(num_classes=len(self.label_map))
                    self.name = f"{trait_name} ({head_type})"
                    self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                    path = f"Training/MODELS/language/AlternativeDataTest/CNNNotReducedNoNer/model"


                case _:
                    raise KeyError(f"Unknown head type: {head_type}")
            
            # Custom State Dict
            if train == 'LoRA':
                self.model = PeftModel.from_pretrained(self.model, path)
                self.model.merge_adapter()
            else:
                print(f"{self.name} loading state dict")
                from safetensors.torch import load_file
                self.model.load_state_dict(load_file(f"{path}/model.safetensors"))

        # === Prepare Model for Inference ===
        self.model.to(DEVICE)
        self.model.eval()

        # === Shap Explainer ===
        self.explainer = shap.Explainer(
            self.predict,
            shap.maskers.Text(self.tokenizer)
        )

        print(f"Initialized model {self.name}")

    def predict(self, text):

        # === Preprocess Input ===
        text = [str(t) for t in text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # === Inference ===
        with torch.no_grad():
            output = self.model(**inputs)["logits"].cpu()

        # === Customized Output ===
        match self.type:
            case 'Classification' | 'MixedCNN' | 'CNN':
                output = torch.softmax(output, dim=-1)
            # case 'classreg':
            #     output = output.squeeze(-1)
            
        return output.numpy()

    def prediction_string(self, text):

        # === String Base ===
        prediction_string = f"{self.name}: "

        # === Inference ===
        pred = self.predict([text])[0]

        # === Customized String ===
        match self.type:
            case 'Classification' | 'MixedCNN' | 'CNN':
                labels = np.argsort(pred)[::-1]
                probs = pred[labels]
                last_index = np.searchsorted(np.cumsum(probs), 0.5) + 1
                for i in range(last_index):
                    prediction_string += f"{self.label_map[labels[i]]} ({100 * probs[i]:.2f}%) "
            case 'Regression':
                if self.label_map == None:
                    prediction_string += f"{pred:.2f}"
                else:
                    prediction_string += f"{self.label_map[round(pred)]} ({pred:.2f})"
        return prediction_string
    
    def explanation_string(self, text):

        # === Inference ===
        shap_values = self.explainer([text])

        # === Customized Explanation ===
        tokens = shap_values.data[0]
        match self.type:
            case 'Classification' | 'MixedCNN' | 'CNN':
                pred = np.argmax(self.predict([text])[0])
                values = shap_values.values[0][:,pred]
            case _:
                values = shap_values.values[0]

        max_blue = min(values)
        max_red = max(values)

        explanation_string = "".join(["\x1b[48;2;" + str(max(round(255 * value / max_red), 0)) + ";0;" + str(max(round(255 * value / max_blue), 0)) + "m" + token 
                                      for token, value in zip(tokens, values)]) + "\x1b[0m"

        return explanation_string
    