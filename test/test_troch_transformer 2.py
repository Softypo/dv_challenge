import transformers as transformers
import torch.nn as nn
# import torch
import numpy as np


def dv_importFromVolume(filepath, RawVol=True, T=True):

    try:
        dtype = np.uint16 if RawVol else np.uint8
        with open(filepath, 'rb') as f:
            RAW = f.read
            data = np.frombuffer(RAW(),
                                 dtype=dtype,
                                 offset=0).reshape(1280, 768, 768)
            if T:
                values = data.T
            else:
                values = data

            return values

    except IOError:
        print("Error while opening the file!")


model = transformers.BertModel.from_pretrained('bert-base-uncased')


class TransformerModel(nn.Module):
    def __init__(self, transformer_model):
        super(TransformerModel, self).__init__()
        self.transformer = transformer_model

    def forward(self, x):
        hidden_states, pooled_output = self.transformer(x)
        return hidden_states, pooled_output


transformer_model = TransformerModel(model)

fc = nn.Linear(768, 3)


def predict_3d_object(x):
    hidden_states, pooled_output = transformer_model(x)
    output = fc(pooled_output)
    return output


scan = dv_importFromVolume(
    "..\data\\training\\volumes\scan_005.raw", T=True)

result = predict_3d_object(scan)

print(result)
