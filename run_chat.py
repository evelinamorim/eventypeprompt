
from transformers import pipeline
import os
import json
import time

model_id = "/projects/F202500017AIVLABDEUCALION/evelinamorim/hf_cache/Qwen3-8B/"

chat_pipeline = pipeline(
    "text-generation",  # Qwen usa text-generation para chat
    model=model_id,
    device_map="auto"
)

input_data = "/projects/F202500017AIVLABDEUCALION/evelinamorim/jsonlusa/"
output_dir ="/projects/F202500017AIVLABDEUCALION/evelinamorim/results/qwen3_8b/"

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

         Now consider the sentence '{sentence_text}', and the following events in the sentence {event_text_lst}. 

         Return only a valid json between <json> and </json>, without explanation.
         
         For instance, the sentence 'A vítima estava presa.', and the events ['estava']. You should produce:
         
         <json>
           {{
              'estava':'STATE'
           }}
         </json>

        """

        # print("Prompt:\n", prompt)

        response = chat_pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False)

        # Optional: extract only the model's answer (strip the prompt if needed)
        output_file = os.path.join(output_dir,file_name)
        with open(output_file, "a") as fd:
            fd.write(response)
        print(f"Model response time {time.time() - start_time}")
        print(50 * "-")
        print()
    break


