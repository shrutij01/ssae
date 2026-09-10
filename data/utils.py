def process_output(og_pred):
    if "</think>" in og_pred:
        reasoning = og_pred.split("</think>")[0] + "</think>"
        answer = og_pred.split("</think>")[1]
    else:
        reasoning = og_pred
        answer = "No think tags found."
    return reasoning.strip(), answer.strip()