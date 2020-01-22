import json
import numpy as np
import seaborn
import matplotlib.pyplot as plt
examples = []
with open("attention_weights_visual.json", 'r') as f:
    for l in f:
        examples.append(json.loads(l))

with open('attention_map_normalized.json', 'w') as f:
    for j, example in enumerate(examples):
        # for y in [0]:
        # for u in [0]:
        # visualize_example = examples[74][0]
        visualize_example = example[0]
        print(j, visualize_example[0])

        title_tokens = [x for x in visualize_example[0].split()
                        if x != "<BOS>"]

        abstract_tokens = visualize_example[1].split()

        heat_map_data = np.array(visualize_example[3])

        heat_map_data = heat_map_data - \
            np.min(heat_map_data, axis=1, keepdims=True)

        def concat_tokens(token_list, heat_map_data, axis):
            while any([x[-2:] == "@@" for x in token_list]):
                for i, t in enumerate(token_list):
                    if t[-2:] == "@@":
                        c = i + 1
                        concat_range = list(range(len(token_list)))
                        while "@@" in token_list[c - 1]:
                            concat_range.remove(c)
                            c += 1
                        new_token_list = token_list[:i] + \
                            ["".join(token_list[i:c])] + token_list[c:]
                        token_list = new_token_list
                        heat_map_data = np.add.reduceat(
                            heat_map_data, concat_range, axis=axis)
                        break
            return token_list, heat_map_data

        title_tokens, heat_map_data = concat_tokens(
            title_tokens, heat_map_data, axis=0)
        abstract_tokens, heat_map_data = concat_tokens(
            abstract_tokens, heat_map_data, axis=1)

        title_tokens = [x.replace("@@", "") for x in title_tokens]
        abstract_tokens = [x.replace("@@", "")
                           for x in abstract_tokens if x != "<PAD>"][1:-1]

        title_length = len(
            title_tokens)

        abstract_length = len(abstract_tokens)

        heat_map_data_no_padding = heat_map_data[:, 1:abstract_length + 1]
        heat_map_data_no_padding = heat_map_data_no_padding - \
            np.min(heat_map_data_no_padding, axis=1, keepdims=True)

        heat_map_data_no_padding = heat_map_data_no_padding / \
            heat_map_data_no_padding.sum(axis=1, keepdims=True)

        attention_map = dict()
        attention_map['title'] = title_tokens
        attention_map['abstract'] = abstract_tokens
        attention_map['attention weights'] = heat_map_data_no_padding.tolist()
        f.write(json.dumps(attention_map) + "\n")

        # seaborn.heatmap(heat_map_data_no_padding, xticklabels=abstract_tokens,
        #                 yticklabels=title_tokens, cmap='inferno_r')

        # plt.show()
