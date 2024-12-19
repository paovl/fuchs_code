import torch
import torchvision.models as models

path = 'resnet18_self_supervised_fold.pth'

# Cargar el archivo .pth
checkpoint = torch.load(path)

# Cargar el modelo ResNet-18 (o cualquier otro modelo según lo que contenga el .pth)
model = models.resnet18()

# Cargar los pesos en el modelo
model.load_state_dict(checkpoint)

# Opcional: Mostrar el modelo o sus parámetros
print(model)