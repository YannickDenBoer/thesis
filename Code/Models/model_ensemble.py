from utils import CUE_LIST
import torch
import os
import cv2
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import gc
from prompts import ensemble_question_list

class SentenceModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def forward_pass(self, sentence: str):
        embedding = self.model.encode(sentence)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        return embedding

# Model functions (forward_pass, generate)
class Model:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def forward_pass(self, answer: str):
        message = [
            { 
                "content": answer
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
    def __init__(self, vl_model, sentence_model):
        self.vl_model = vl_model
        self.sentence_model = sentence_model

    def create_answer_embeddings(self, frame: Image.Image):
        """
        For each question and each of its templates, generate the model answer,
        embed it, and collect per‐template embeddings, with error handling.
        Returns: Tensor of shape (num_questions, num_templates, embedding_dim)
        """
        all_question_embeds = []
        # determine embedding dimension once
        #embed_dim = self.sentence_model.model.get_sentence_embedding_dimension()
        embed_dim = self.vl_model.model.config.hidden_size

        for qdict in ensemble_question_list:
            template_embeds = []
            for template in qdict["templates"]:
                try:
                    # generate and embed
                    gen_text = self.vl_model.generate(template, frame)
                    text_embedding = self.vl_model.forward_pass(gen_text)
                    #text_embedding = self.sentence_model.forward_pass(f"{template} [SEP] {gen_text}")
                except Exception as e:
                    print(f"Error processing template '{template}': {e}")
                    # fallback to NaNs of correct size
                    text_embedding = torch.full((embed_dim,), float("nan"))
                # collect and clean up
                template_embeds.append(text_embedding)
                if "gen_text" in locals():
                    del gen_text
                torch.cuda.empty_cache()

            # stack per‐template embeddings: (num_templates, D)
            all_question_embeds.append(torch.stack(template_embeds))
            torch.cuda.empty_cache()

        # final stack: (num_questions, num_templates, D)
        all_question_embeds = torch.stack(all_question_embeds)
        print(all_question_embeds.shape)
        return all_question_embeds

    def images_to_embeddings(self, X):
        all_samples = []
        for i, x in enumerate(X):
            print(f"Processing image {i+1}/{len(X)}")
            print(x.shape, x.dtype, x.min(), x.max(), np.isnan(x).any(), np.isinf(x).any())
            frame = Image.fromarray(x)
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            answer_embeddings = self.create_answer_embeddings(frame)

            all_samples.append(answer_embeddings)
            del answer_embeddings
            torch.cuda.empty_cache()
        # convert answer_embeddings to tensor
        all_samples = torch.stack(all_samples) # shape (num_samples, num_questions, num_templates, embedding_dim)
        gc.collect()
        return all_samples
    
    def true_answers_to_embeddings(self):
        true_answer_embeddings = []
        for qdict in ensemble_question_list:
            template_embeds = []
            for question, answer in zip(qdict["templates"], qdict["pos_paraphrases"]):
                #true_embed = self.sentence_model.forward_pass(f"{question} [SEP] {answer}")
                true_embed = self.vl_model.forward_pass(answer)
                template_embeds.append(true_embed)
                del true_embed
                torch.cuda.empty_cache()
            true_answer_embeddings.append(torch.stack(template_embeds))
        true_answer_embeddings = torch.stack(true_answer_embeddings)
        return true_answer_embeddings # shape (num_questions, Num templates, embedding_dim)
        
    def embeddings_to_cos_scores(self, answer_embeddings: torch.Tensor,
                              true_answer_embeddings: torch.Tensor,
                              pool: str = "mean"):  # 'mean' or 'max'
        """
        Compute one-to-one cosine similarity between each pred-template and the corresponding true-paraphrase,
        then pool across the template axis.

        answer_embeddings: (N, Q, T, D)
        true_answer_embeddings: (Q, T, D)
        Returns: Tensor of shape (N, Q) of pooled scores
        """
        # Align dims: true -> (1, Q, T, D)
        true_exp = true_answer_embeddings.unsqueeze(0)
        # Compute cos per template: (N, Q, T)
        cos_pair = F.cosine_similarity(answer_embeddings, true_exp, dim=3)
        # Pool over templates
        if pool == "mean":
            return cos_pair.mean(dim=2)
        elif pool == "max":
            return cos_pair.max(dim=2).values
        else:
            raise ValueError(f"Unknown pool type '{pool}'")

    def embeddings_to_cos_scores5(
        answer_embeddings: torch.Tensor,     # (N, Q, T, D)
        true_answer_embeddings: torch.Tensor # (Q, T, D)
    ) ->     torch.Tensor:
        N, Q, T, D = answer_embeddings.shape

        # 1) Collapse the template dimension into the embedding axis:
        #    -> ans_concat: (N, Q, T*D)
        #    -> true_concat: (1, Q, T*D)
        ans_concat  = answer_embeddings.view(N, Q, T * D)
        true_concat = true_answer_embeddings.view(1, Q, T * D)

        # 2) Cosine along the last axis gives (N, Q)
        cos_scores = F.cosine_similarity(ans_concat, 
                                         true_concat, 
                                         dim=2)
        return cos_scores