from utils import CUE_LIST
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import gc

question_list = [("label, pos_answer, neg_answer")]

# Model functions (forward_pass, generate)
class Model:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def forward_pass(self, frame, question: str):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad(): 
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        # Extract embedding from the last layer's last token
        seq_len = inputs.attention_mask[0].sum().item()
        embedding = outputs.hidden_states[-1][0, seq_len-1].to(torch.float32).detach().cpu()

        mask = inputs.attention_mask[0].unsqueeze(-1)  # shape [T, 1]
        #element-wise mask en optellen over tokens
        sum_embeds = (outputs.hidden_states[-1][0] * mask).sum(dim=0)   # shape [D]
        mean_embedding = (sum_embeds / seq_len).to(torch.float32).detach().cpu()

        return embedding

    def generate(self, question: str, frame: Image.Image):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

        # Generate output and Decode output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0]
    
class Processor:
    def __init__(self, vl_model):
        self.vl_model = vl_model

    def create_answer_embeddings(self, frame, question_list):        
        embeddings = [] # shape (num_questions, embedding_dim)
        for label, question, pos_answer, neg_answer in question_list:
            try:
                pred_embed = self.vl_model.forward_pass(frame, question)
                #print(f"question: {question}")
                #print(f"Generated text: {gen_text}")
                #pred_embed = self.sentence_model.forward_pass(f"{question} [SEP] {gen_text}")
                
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                # fallback to zero vector of correct shape
                vl_model_dim = self.vl_model.model.config.hidden_size
                #sentence_model_dim = self.sentence_model.model.get_sentence_embedding_dimension()
                pred_embed = torch.full(
                (vl_model_dim,), float('nan')
                )
                
            embeddings.append(pred_embed)
            del pred_embed
            torch.cuda.empty_cache()
        embeddings = torch.stack(embeddings)
        return embeddings
    
    def images_to_embeddings(self, X):
        answer_embeddings = [] # shape (num_samples, num_questions, embedding_dim)
        for i, x in enumerate(X):
            print(f"Processing image {i+1}/{len(X)}")
            print(x.shape, x.dtype, x.min(), x.max(), np.isnan(x).any(), np.isinf(x).any())
            frame = Image.fromarray(x)
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            embedding = self.create_answer_embeddings(frame, question_list)

            answer_embeddings.append(embedding)
            del embedding
            torch.cuda.empty_cache()
        # convert answer_embeddings to tensor
        answer_embeddings = torch.stack(answer_embeddings) # shape (num_samples, num_questions, embedding_dim)
        gc.collect()
        return answer_embeddings