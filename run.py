from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

import os
import json
import time

#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "/projects/F202500017AIVLABDEUCALION/evelinamorim/hf_cache/Qwen3-8B/"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
     model_id,
     torch_dtype=torch.bfloat16, # ou torch.float16, dependendo do modelo e hardware
     device_map="auto",
     # load_in_4bit=True # Descomente para carregamento quantizado se necessário
)

input_data = "/projects/F202500017AIVLABDEUCALION/evelinamorim/jsonlusa/"
output_dir ="/projects/F202500017AIVLABDEUCALION/evelinamorim/results/quen3_8b/"

file_name_lst = os.listdir(input_data)
for file_name in file_name_lst:
    json_data = os.path.join(input_data, file_name)
    with open(json_data, 'r') as fd:
        data = json.load(fd)

    for sentence in data["sentences"]:
        start_time = time.time()
        event_lst = []

        for event in sentence["events"]:
            event_type = event["Event_Type"] if "Event_Type" in event else None
            event_lst.append((event["text"], event_type))

        if event_lst == []:
            continue
        sentence_text = sentence["text"]

        event_text_lst = [event[0] for event in event_lst]

        prompt = f"""
           You are an experienced Portuguese linguist in data annotation. Your task is to identify the type of event in the Portuguese language,
           where an event can be defined as an eventuality that happens or occurs, or a state or circumstance that is temporally relevant,
           that is, that is directly related to a temporal expression or change throughout the text. The types of events are:

           1) STATE – situation in which something obtains or holds true. Ex. A vítima estava
          presa.
          2) PROCESS – situation that is dynamic and atelic, with duration. Ex. O João
          nadou.
          3) TRANSITION – situation that is dynamic, with duration and consequent state. It
         subsumes accomplishments and achievements. Ex. O João leu um livro./ O João
          ganhou a corrida.
         4) None - there is no suitable type for the event.

         Now consider the sentence {sentence_text}, and the following events in the sentence {event_text_lst}. 

         Return only a json, nothing more, which has each event as a key, and its value as a dictionary with the type value and the justification for the type.

        """

        # print("Prompt:\n", prompt)
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # adjust as needed
            temperature=0.7,  # controls creativity
            do_sample=True
        )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optional: extract only the model's answer (strip the prompt if needed)
        output_file = os.path.join(output_dir,file_name)
        with open(output_file, "a") as fd:
            fd.write(response)
        print(f"Model response time {time.time() - start_time}")
        print(50 * "-")
        print()
